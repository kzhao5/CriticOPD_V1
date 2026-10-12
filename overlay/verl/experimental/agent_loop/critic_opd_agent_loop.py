# CriticOPD agent loop: the student writes a full solution; if it is wrong, the teacher critiques it,
# the solution is cut after the located error, and the student rewrites from there with the feedback
# in its context.
#
# Unlike teacher-takeover methods, the teacher never writes part of the solution: it generates the
# critique on a separate critic prompt.
#
# Training sequence of a repaired sample (response_mask):
#   kept prefix -> 0    feedback -> not in the sequence    student rewrite -> 1
# The feedback is only in the context while the rewrite is generated. With CRITIC_OPD_LOSS=fb the
# teacher scores the rewrite with the feedback in its context (see agent_loop.py), while the student
# is trained on the sequence without it, so it has to produce the corrected reasoning on its own.
import asyncio
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_ROOT = str(Path(__file__).resolve().parents[3])        # repository root that contains verl/ and opd/

_SEP = re.compile(r"\n\s*\n")
_LEADIN = re.compile(r"^\s*(?:#{1,6}\s|[-*_=]{3,}\s*$|\*\*[^*]{0,40}\*\*\s*:?\s*$"
                     r"|(?:So|Now|Then|Next|Thus|Hence|Therefore|Step\s*\d+|Case\s*\d+)\b[^.\n]{0,30}:\s*$)", re.I)
_RE_SEG = re.compile(r"SEGMENT:\s*\[?(\d+)", re.I)
_RE_Q = re.compile(r"QUOTE:\s*(.+?)(?=\n\s*FEEDBACK:|\Z)", re.I | re.S)
_RE_FB = re.compile(r"FEEDBACK:\s*(.+)", re.I | re.S)

# ---- single-error critic without the reference answer (ablation: CRITIC_OPD_GIVE_GT=0) ----
CRITIC_SYS = (
    "You are a meticulous math teacher reviewing a student's solution. "
    "The student's FINAL ANSWER IS WRONG. Find the FIRST place where the reasoning goes wrong.\n\n"
    "Rules:\n"
    "1. Identify exactly ONE segment -- the FIRST containing a genuine error.\n"
    "2. Quote the exact wrong sentence verbatim.\n"
    "3. Say briefly what is wrong and what to do instead.\n"
    "4. DO NOT continue the solution. DO NOT compute or state the final answer.\n\n"
    "Reply exactly as:\nSEGMENT: <index>\nQUOTE: <verbatim>\nFEEDBACK: <what is wrong and what to do>"
)
# ---- single-error critic with the reference answer (CRITIC_OPD_GIVE_GT=1) ----
# The critic may reply NONE: a wrong location is worse than none, because the student would rewrite
# from a step that was correct.
_GT_RULES = (
    "How to locate the error:\n"
    "Read the numbered segments in order. For each one ask: given everything before it, is this "
    "step still correct, and does it still allow the problem to be solved? The FIRST segment "
    "where the answer is no is the error.\n\n"
    "Do NOT flag a segment merely because it is: an exploratory attempt the student later "
    "abandons or corrects; a longer route than necessary; or sloppy notation that does not change "
    "the mathematics. Flag only a step that is actually wrong.\n\n"
    "Rules you must follow:\n"
    "1. Identify exactly ONE segment -- the FIRST one containing a genuine error.\n"
    "2. Quote verbatim the single sentence in that segment that carries the error.\n"
    "3. Feedback: say what is wrong and what to do from that point instead. The student will "
    "resume writing immediately after this point, so phrase it as guidance they can act on. "
    "Two or three sentences at most.\n"
    "4. NEVER reveal the final answer. Do not state it, do not write an expression that "
    "evaluates to it, do not say what it should be. Describe the method, not the result.\n"
    "5. If every step looks correct to you and you cannot identify a genuine error, reply with "
    "SEGMENT: NONE. Do not guess -- a wrong location is worse than none.\n\n"
    "Reply in exactly this format:\n"
    "SEGMENT: <index, or NONE>\n"
    "QUOTE: <the exact wrong sentence, copied verbatim>\n"
    "FEEDBACK: <what is wrong and what to do instead>"
)
CRITIC_SYS_GT = (
    "You are a meticulous math teacher. A student has submitted a complete solution whose "
    "FINAL ANSWER IS WRONG. You are given the correct answer. Use it as a reference to trace the "
    "student's work forward and find the EARLIEST step after which the correct answer can no "
    "longer be reached.\n\n" + _GT_RULES)
# A rollout that hits the length limit has no final answer: the critic decides whether it went off track.
CRITIC_SYS_GT_CAPPED = (
    "You are a meticulous math teacher. A student's solution ran on without ever reaching an "
    "answer. You are given the correct answer. Use it as a reference to trace the student's work "
    "forward and find the EARLIEST step that sent the reasoning off track. If the work is still on "
    "a valid path and was merely slow, reply SEGMENT: NONE.\n\n" + _GT_RULES)
_RE_NONE = re.compile(r"SEGMENT:\s*\[?\s*(NONE|N/A|-)\b", re.I)

# ---- ablation CRITIC_OPD_TRUNC=1: keep only the first paragraph of the rewrite ----
# The boundary is the first blank line after non-blank content.
_BLANK = re.compile(r"\S[^\S\n]*\n[^\S\n]*\n")


def _cut_at_blank_line(tok, ids: list[int]) -> int:
    """Return k such that ids[:k] ends at the first blank line after non-blank content (inclusive).
    Whether a prefix contains such a blank line is monotone in k, so binary search; no blank line -> keep all."""
    if not ids or not _BLANK.search(tok.decode(ids, skip_special_tokens=True)):
        return len(ids)
    lo, hi = 1, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if _BLANK.search(tok.decode(ids[:mid], skip_special_tokens=True)):
            hi = mid
        else:
            lo = mid + 1
    return lo


FEEDBACK_TMPL = ("\n\n[Teacher feedback] {fb}\n\nUsing this feedback, continue the solution from here. "
                 "Put your final answer in \\boxed{{}}.\n\n")


def _is_leadin(p):
    return bool(_LEADIN.match(p)) or (len(p) <= 12 and p.rstrip().endswith(":"))


def segment(text):
    """Split at blank lines, then merge headings, separators and lead-in lines into the next segment.
    Merging by form rather than by length keeps short content blocks (one formula, one conclusion) separate."""
    parts = [p.strip() for p in _SEP.split(text) if p.strip()]
    merged, buf = [], []
    for p in parts:
        if _is_leadin(p):
            buf.append(p); continue
        merged.append("\n\n".join(buf + [p]) if buf else p); buf = []
    if buf: merged.append("\n\n".join(buf))
    return merged or parts


def parse_critic(text, segs):
    """Locate the error by its quote in the student text; the segment index is only a fallback,
    since the critic's index is often off by one or two while the quote is not."""
    if _RE_NONE.search(text):          # the critic found no error: do not let the quote match override it
        return None, None
    mq, mf, ms = _RE_Q.search(text), _RE_FB.search(text), _RE_SEG.search(text)
    fb = mf.group(1).strip() if mf else None
    quote = mq.group(1).strip().strip('"') if mq else None
    idx = None
    if quote and len(quote) >= 8:
        norm = lambda t: re.sub(r"\s+", " ", t).strip()
        nq = norm(quote)[:80]
        for i, s in enumerate(segs):
            if quote[:60] in s or nq in norm(s):
                idx = i; break
    if idx is None and ms:
        j = int(ms.group(1))
        if 0 <= j < len(segs): idx = j
    return idx, fb


# ---- multi-error critic (CRITIC_OPD_KERR): list the errors in order and cut after the k-th ----
# Same rules as CRITIC_SYS_GT, except that the critic lists all errors (at most 5) instead of the first.
_MULTI_RULES = (
    "How to locate the errors:\n"
    "Read the numbered segments in order. For each one ask: given everything before it, is this step "
    "correct? List every segment that contains a genuine error, in the order in which they appear.\n\n"
    "Do NOT flag a segment merely because it is: an exploratory attempt the student later abandons or "
    "corrects; a longer route than necessary; or sloppy notation that does not change the mathematics. "
    "Flag only a step that is actually wrong.\n\n"
    "Rules you must follow:\n"
    "1. List the erroneous segments in order of appearance, at most 5.\n"
    "2. For each one, quote verbatim the single sentence that carries the error.\n"
    "3. For each one, say what is wrong and what to do instead. The student will resume writing after "
    "the last listed error, so phrase it as guidance they can act on. Two or three sentences per error.\n"
    "4. NEVER reveal the final answer. Do not state it, do not write an expression that evaluates to it, "
    "do not say what it should be. Describe the method, not the result.\n"
    "5. If every step looks correct to you and you cannot identify a genuine error, reply with "
    "SEGMENT: NONE. Do not guess -- a wrong location is worse than none.\n\n"
    "Reply in exactly this format, one block per error:\n"
    "ERROR 1\nSEGMENT: <index>\nQUOTE: <the exact wrong sentence, copied verbatim>\n"
    "FEEDBACK: <what is wrong and what to do instead>\n"
    "ERROR 2\nSEGMENT: <index>\nQUOTE: <...>\nFEEDBACK: <...>")
CRITIC_SYS_MULTI = (
    "You are a meticulous math teacher. A student has submitted a complete solution whose FINAL ANSWER "
    "IS WRONG. You are given the correct answer. Use it as a reference to trace the student's work "
    "forward and find the steps that contain genuine errors.\n\n" + _MULTI_RULES)
CRITIC_SYS_MULTI_CAPPED = (
    "You are a meticulous math teacher. A student's solution ran on without ever reaching an answer. "
    "You are given the correct answer. Use it as a reference to trace the student's work forward and find "
    "the steps that sent the reasoning off track. If the work is still on a valid path and was merely "
    "slow, reply SEGMENT: NONE.\n\n" + _MULTI_RULES)
# Ablation CRITIC_OPD_FB_MODE=generic: same template, feedback without content.
GENERIC_FEEDBACK = "Your solution contains errors. Check the previous reasoning and redo it."
# Ablation CRITIC_OPD_ALL=1: a correct rollout cannot be called wrong, so the first sentence is neutral.
CRITIC_SYS_MULTI_ANY = (
    "You are a meticulous math teacher. A student has submitted a complete solution. You are given the correct "
    "answer. Use it as a reference to trace the student's work forward and find the steps that contain genuine "
    "errors.\n\n" + _MULTI_RULES)
_RE_ERRHDR = re.compile(r"\bERROR\s*\d+\s*:?\s*$", re.I)


def parse_critic_multi(text, segs, max_err=5):
    """Parse a multi-error critique into [(segment index, quote, feedback)], sorted by index, deduplicated.
    Blocks are split at "SEGMENT:" and located with parse_critic; a trailing "ERROR k" header is removed
    from the feedback. Returns [] if the critique only says NONE."""
    errs, seen = [], set()
    for b in re.split(r"(?i)(?=SEGMENT\s*:)", text):
        if not re.match(r"(?i)SEGMENT\s*:", b):
            continue
        idx, fb = parse_critic(b, segs)
        if idx is None or not fb or idx in seen:
            continue
        fb = "\n".join(ln for ln in fb.strip().splitlines() if not _RE_ERRHDR.search(ln)).strip()
        if not fb:
            continue
        mq = _RE_Q.search(b)
        q = mq.group(1).strip().strip('"') if mq else ""
        errs.append((idx, q, fb)); seen.add(idx)
    errs.sort(key=lambda e: e[0])
    return errs[:max_err]


_RE_LAST_BLOCK = re.compile(r"(?i)(?:ERROR\s*\d+\s*:?\s*)?SEGMENT\s*:")


def drop_incomplete_tail(text):
    """If the critique hit its length limit, its last block (from the last SEGMENT:) is incomplete: drop it.
    Only called on truncated critiques; text without SEGMENT: is returned unchanged."""
    ms = list(_RE_LAST_BLOCK.finditer(text))
    return text[:ms[-1].start()] if ms else text


def kerr_select(errs, kerr):
    """Cut after the k-th error, or after the last one if there are fewer than k. kerr is '1', '2', ... or 'last'."""
    if not errs:
        return []
    k = len(errs) if str(kerr) == "last" else min(int(kerr), len(errs))
    return errs[:k]


def leaks_answer(fb, gt, stu):
    """The feedback contains the reference answer as a separate token while the student text does not."""
    if gt is None:
        return False
    g = str(gt).strip().strip("$")
    if not g:
        return False
    # no adjacent letter/digit or decimal part (3280, 1.328, 23.5 do not match 328 / 23); a sentence-final "23." does
    pat = r"(?<![\w.])" + re.escape(g) + r"(?!\w|\.\d)"
    return bool(re.search(pat, fb or "")) and not re.search(pat, stu or "")


def drop_leaky(errs, gt, stu):
    """Drop the errors whose feedback states the reference answer (used when CRITIC_OPD_FB_ANSWER=0)."""
    return [e for e in errs if not leaks_answer(e[2], gt, stu)]


def kerr_feedback(sel):
    """One error: its feedback as is. Several errors: listed in order with their quotes (cut to 200 characters)."""
    if len(sel) == 1:
        return sel[0][2]
    parts = [f'({i + 1}) In "{q[:200]}": {fb}' for i, (_, q, fb) in enumerate(sel)]
    return f"Your solution contains {len(sel)} errors.\n" + "\n".join(parts)


def _tok_prefix_for_text(tok, ids: list[int], text: str) -> int:
    """Smallest k such that decode(ids[:k]) covers text.

    The kept prefix is cut on the original rollout tokens rather than re-encoded from text, so the training
    sequence stays a prefix of the student's own trajectory and its rollout log-probs stay aligned.
    """
    target = len(text.rstrip())
    if target <= 0:
        return 0
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tok.decode(ids[:mid], skip_special_tokens=True).rstrip()) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ---- answer grading in long-lived subprocesses ----
# A grader call can hold the GIL for a long time (e.g. sympy evaluating a huge power), which would block the
# agent loop's event loop and even its timeouts. Grading therefore runs in subprocesses that answer over
# stdin/stdout; on timeout the subprocess is killed and replaced.
_GRADER_SRC = r"""
import sys, os, json
out = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)                         # library prints go to stderr; the protocol uses `out` only
sys.path.insert(0, os.environ["CRITIC_OPD_ROOT"])
os.environ.setdefault("MATH_GRADER_PATH", os.path.join(os.environ["CRITIC_OPD_ROOT"], "opd", "reward", "grader"))
from opd.reward.math_reward import compute_score
out.write("@@READY\n")
for line in sys.stdin:
    try:
        q = json.loads(line)
        v = compute_score(solution_str=q["t"], ground_truth=q["g"])
        r = float(v["score"] if isinstance(v, dict) else v) > 0.5
    except Exception:
        r = False
    out.write("@@R " + ("1" if r else "0") + "\n")
"""


class _GraderPool:
    _pools: dict = {}

    def __init__(self, n: int):
        self.n = n
        self.idle: asyncio.Queue = asyncio.Queue()
        self.started = False
        self.lock = asyncio.Lock()

    @classmethod
    def get(cls) -> "_GraderPool":
        loop = asyncio.get_running_loop()
        p = cls._pools.get(id(loop))
        if p is None:
            p = cls._pools[id(loop)] = cls(int(os.environ.get("CRITIC_OPD_GRADER_PROCS", "8")))
        return p

    async def _spawn(self):
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", _GRADER_SRC, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            env={**os.environ, "CRITIC_OPD_ROOT": _ROOT})
        while True:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=180)
            if not line:
                raise RuntimeError("grader subprocess failed to start")
            if line.startswith(b"@@READY"):
                return proc

    async def _replace(self, proc):
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        asyncio.ensure_future(proc.wait())            # reap, no zombie
        for _ in range(5):
            try:
                self.idle.put_nowait(await self._spawn())
                return
            except Exception as e:                    # noqa: BLE001
                logger.warning("[critic_opd] failed to respawn a grader subprocess: %s", e)
                await asyncio.sleep(5)

    async def _ensure(self):
        if self.started:
            return
        async with self.lock:
            if self.started:
                return
            for proc in await asyncio.gather(*[self._spawn() for _ in range(self.n)]):
                self.idle.put_nowait(proc)
            self.started = True

    async def grade(self, text: str, gt, timeout: float) -> bool:
        await self._ensure()
        proc = await self.idle.get()
        try:
            proc.stdin.write((json.dumps({"t": text, "g": str(gt)}) + "\n").encode())
            await proc.stdin.drain()

            async def _read():
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        raise RuntimeError("grader subprocess exited")
                    if line.startswith(b"@@R "):
                        return line[4:5] == b"1"

            r = await asyncio.wait_for(_read(), timeout=timeout)
        except BaseException:
            # timeout / cancellation / dead subprocess: a late answer could be read by the next request,
            # so the subprocess is always replaced.
            asyncio.ensure_future(self._replace(proc))
            raise
        self.idle.put_nowait(proc)
        return r


@register("critic_opd_agent")
class CriticOpdAgentLoop(AgentLoopBase):
    # Class-level defaults, so that every switch is defined even if __init__ is bypassed.
    give_gt = False
    trunc = False
    trunc_max = 512
    keep_correct = False
    critic_maxlen = 34817
    critic_tokens_exact = False
    fb_answer = False
    kerr = None
    prefix_loss = False
    critic_all = False
    cut_mode = "error"
    fb_mode = "critique"
    anneal = None
    global_step = -1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        # where the rewrite starts: "after" keeps the erroneous segment in the prefix, "before" drops it
        self.splice = os.environ.get("CRITIC_OPD_SPLICE", "before").strip().lower()
        self.enabled = os.environ.get("CRITIC_OPD_ENABLE", "1") == "1"
        # 0: rewrite from the error without feedback
        self.use_feedback = os.environ.get("CRITIC_OPD_USE_FEEDBACK", "1") == "1"
        self.critic_max_tokens = int(os.environ.get("CRITIC_OPD_CRITIC_TOKENS", "512"))
        # the critic receives the reference answer
        self.give_gt = os.environ.get("CRITIC_OPD_GIVE_GT", "0") == "1"
        # keep only the first paragraph of the rewrite (at most TRUNC_MAX tokens)
        self.trunc = os.environ.get("CRITIC_OPD_TRUNC", "0") == "1"
        self.trunc_max = int(os.environ.get("CRITIC_OPD_TRUNC_MAX", "512"))
        # keep only rewrites with a correct final answer; otherwise train on the original rollout as in OPD
        self.keep_correct = os.environ.get("CRITIC_OPD_KEEP_CORRECT", "0") == "1"
        self.critic_maxlen = int(os.environ.get("CRITIC_OPD_CRITIC_MAXLEN", "34817"))   # teacher max_model_len
        # the multi-error critic uses a limit of at least 1024 tokens; =1 uses CRITIC_OPD_CRITIC_TOKENS exactly
        self.critic_tokens_exact = os.environ.get("CRITIC_OPD_CRITIC_TOKENS_EXACT", "0") == "1"
        # append the reference answer to the feedback (and do not filter feedback that states it)
        self.fb_answer = os.environ.get("CRITIC_OPD_FB_ANSWER", "0") == "1"
        # multi-error critic, cut after the k-th error: '1' / '2' / '3' / ... / 'last'; unset = single-error critic
        self.kerr = (os.environ.get("CRITIC_OPD_KERR", "").strip().lower() or None)
        # also apply the OPD loss to the student tokens before the first error
        self.prefix_loss = os.environ.get("CRITIC_OPD_PREFIX_LOSS", "0") == "1"
        # critique every rollout, including correct ones (with a neutral critic prompt)
        self.critic_all = os.environ.get("CRITIC_OPD_ALL", "0") == "1"
        # CUT=end keeps the whole incorrect solution (rollouts that hit the length limit are still cut at the
        # error, otherwise there is no room to continue); FB_MODE=generic replaces the critique with
        # GENERIC_FEEDBACK at the same cut point (the teacher also scores with it)
        self.cut_mode = os.environ.get("CRITIC_OPD_CUT", "error").strip().lower()
        self.fb_mode = os.environ.get("CRITIC_OPD_FB_MODE", "critique").strip().lower()
        # phase-out schedule "start,end,min": an incorrect rollout is repaired with probability beta, which
        # falls linearly from 1 at step `start` to `min` at step `end`; ":" also works as a separator
        _a = os.environ.get("CRITIC_OPD_ANNEAL", "").strip()
        self.anneal = tuple(float(x) for x in _a.replace(":", ",").split(",")) if _a else None
        self.global_step = int(kwargs.get("critic_global_step", -1) or -1)

    def _beta(self) -> float:
        if not self.anneal or self.global_step < 0:
            return 1.0
        t0, t1, bmin = self.anneal
        if self.global_step <= t0:
            return 1.0
        if self.global_step >= t1:
            return bmin
        return 1.0 - (1.0 - bmin) * (self.global_step - t0) / max(t1 - t0, 1e-9)

    async def _verify_async(self, text, gt) -> bool:
        """Check whether a rewrite is correct. Timeout or error counts as incorrect (the rewrite is not kept)."""
        try:
            return bool(await asyncio.wait_for(_GraderPool.get().grade(text, gt, 30), timeout=240))
        except (TimeoutError, asyncio.TimeoutError, RuntimeError, OSError) as e:
            logger.warning("[critic_opd] verifying the rewrite failed (%s), counted as incorrect", type(e).__name__)
            return False

    async def _grade_async(self, text, gt):
        """Decide whether to critique. Timeout or error counts as correct (no critique)."""
        try:
            return bool(await asyncio.wait_for(_GraderPool.get().grade(text, gt, 30), timeout=240))
        except (TimeoutError, asyncio.TimeoutError, RuntimeError, OSError) as e:
            logger.warning("[critic_opd] grading failed (%s), counted as correct", type(e).__name__)
            return True

    async def _teacher_generate(self, prompt_ids, max_tokens):
        """Greedy free generation by the teacher (the critique) through the teacher client used for scoring."""
        tm = self.teacher_manager
        if tm is None: return None
        key = next(iter(tm.teacher_model_configs))
        out = await tm.teacher_client[key].generate(
            request_id=uuid4().hex, prompt_ids=prompt_ids,
            sampling_params={"max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0},
        )
        return out.token_ids

    _CNT = {"n": 0, "capped": 0, "correct": 0, "attempted": 0, "parsed": 0,
            "repaired": 0, "exc": 0}

    def _tally(self, **kw):
        c = type(self)._CNT
        c["n"] += 1
        for k in kw:
            c[k] = c.get(k, 0) + 1
        if c["n"] % 64 == 0:
            print(f"[critic_opd] tally {c}", flush=True)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        try:
            return await self._run(sampling_params, **kwargs)
        except Exception:
            import traceback
            self._tally(exc=1)
            logger.error("[critic_opd] run() failed, falling back to a plain rollout:\n%s", traceback.format_exc())
            return await self._plain(sampling_params, **kwargs)

    async def _plain(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Fallback: a single-turn rollout without critique or rewrite."""
        messages = list(kwargs["raw_prompt"])
        mm = await self.process_multi_modal_info(messages)
        mmk = self._get_mm_processor_kwargs(mm.get("audios"))
        prompt_ids = await self.apply_chat_template(
            messages, images=mm.get("images"), videos=mm.get("videos"),
            audios=mm.get("audios"), mm_processor_kwargs=mmk)
        out: TokenOutput = await self.server_manager.generate(
            request_id=uuid4().hex, prompt_ids=prompt_ids,
            sampling_params=sampling_params, mm_processor_kwargs=mmk)
        rlen = int(sampling_params.get("max_tokens") or self.response_length)
        resp = out.token_ids[:rlen]
        # response_logprobs must be present for every sample of a batch (they are concatenated)
        _lp = list(out.log_probs[:len(resp)]) if out.log_probs is not None else None
        if _lp is not None and len(_lp) < len(resp):
            _lp += [0.0] * (len(resp) - len(_lp))
        return AgentLoopOutput(
            prompt_ids=prompt_ids, response_ids=resp, response_mask=[1] * len(resp),
            response_logprobs=_lp,
            multi_modal_data=mm, mm_processor_kwargs=mmk, num_turns=2,
            metrics={"num_preempted": -1},
            extra_fields={**(out.extra_fields or {}), "critic_opd": {"fallback": 1}})

    async def _run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        mm = await self.process_multi_modal_info(messages)
        mmk = self._get_mm_processor_kwargs(mm.get("audios"))
        prompt_ids = await self.apply_chat_template(
            messages, images=mm.get("images"), videos=mm.get("videos"),
            audios=mm.get("audios"), mm_processor_kwargs=mmk)

        metrics = {}
        with simple_timer("generate_sequences", metrics):
            out: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex, prompt_ids=prompt_ids,
                sampling_params=sampling_params, mm_processor_kwargs=mmk)
        rlen = int(sampling_params.get("max_tokens") or self.response_length)
        resp = out.token_ids[:rlen]
        stats = {"critic_attempted": 0, "critic_parsed": 0, "repaired": 0}

        gt = None
        rm = kwargs.get("reward_model")
        if isinstance(rm, dict): gt = rm.get("ground_truth")

        # A finished rollout is graded against the reference answer and critiqued only if it is wrong.
        # A rollout that hit the length limit has no answer; with the reference answer the critic decides
        # whether it went off track (it replies NONE if not).
        capped = len(out.token_ids) >= rlen
        text = self.tokenizer.decode(resp, skip_special_tokens=True)
        ready = self.enabled and self.teacher_manager is not None and gt is not None
        is_correct = False
        if capped:
            need = ready and self.give_gt
        else:
            is_correct = bool(ready) and await self._grade_async(text, gt)
            need = ready and (not is_correct or self.critic_all)
            if is_correct and need:
                stats["critic_on_correct"] = 1
                self._tally(correct_critiqued=1)

        if capped:
            self._tally(capped=1)
        elif not need:
            self._tally(correct=1)
        if need and self.anneal is not None:
            _b = self._beta()
            stats["anneal_beta"] = _b
            if random.random() >= _b:
                need = False
                stats["anneal_skip"] = 1
                self._tally(anneal_skip=1)

        if need:
            self._tally(attempted=1)
            stats["critic_attempted"] = 1
            segs = segment(text)
            body = "\n\n".join(f"[{i}] {s}" for i, s in enumerate(segs))
            if self.kerr:
                cmsg = [{"role": "system", "content": CRITIC_SYS_MULTI_CAPPED if capped else (CRITIC_SYS_MULTI_ANY if is_correct else CRITIC_SYS_MULTI)},
                        {"role": "user", "content":
                         f"Problem:\n{messages[-1]['content']}\n\n"
                         f"Correct answer (reference only -- never reveal it): {gt}\n\n"
                         f"Student's solution, split into numbered segments:\n\n{body}"}]
            elif self.give_gt:
                cmsg = [{"role": "system", "content": CRITIC_SYS_GT_CAPPED if capped else CRITIC_SYS_GT},
                        {"role": "user", "content":
                         f"Problem:\n{messages[-1]['content']}\n\n"
                         f"Correct answer (reference only -- never reveal it): {gt}\n\n"
                         f"Student's solution, split into numbered segments:\n\n{body}"}]
            else:
                cmsg = [{"role": "system", "content": CRITIC_SYS},
                        {"role": "user", "content":
                         f"Problem:\n{messages[-1]['content']}\n\nStudent's solution, numbered segments:\n\n{body}"}]
            cids = self.tokenizer.apply_chat_template(cmsg, add_generation_prompt=True, tokenize=False)
            cids = self.tokenizer.encode(cids, add_special_tokens=False)
            _cmax = max(1024, self.critic_max_tokens) if (self.kerr and not self.critic_tokens_exact) else self.critic_max_tokens
            ctok = await self._teacher_generate(cids, _cmax)
            if ctok:
                ctext = self.tokenizer.decode(ctok, skip_special_tokens=True)
                if self.kerr and len(ctok) >= _cmax:
                    # truncated multi-error critique: its last feedback is unfinished, drop that block and
                    # cut after the previous complete error
                    ctext = drop_incomplete_tail(ctext)
                    stats["critic_truncated"] = 1
                    self._tally(kerr_trunc_drop=1)
                if self.kerr:
                    errs = parse_critic_multi(ctext, segs)
                    n_raw = len(errs)
                    _first_raw = errs[0][0] if errs else None   # boundary of the prefix loss: first error found
                    if not self.fb_answer:
                        errs = drop_leaky(errs, gt, text)
                    if n_raw > len(errs):
                        stats["critic_n_leak"] = n_raw - len(errs)
                        self._tally(kerr_leak_drop=n_raw - len(errs))
                    sel = kerr_select(errs, self.kerr)
                    idx, fb = (sel[-1][0], kerr_feedback(sel)) if sel else (None, None)
                    first_idx = _first_raw if sel else None
                    stats["critic_n_err"] = len(errs); stats["critic_k_eff"] = len(sel)
                    if sel:
                        self._tally(**{f"kerr_k{len(sel)}": 1})
                else:
                    idx, fb = parse_critic(ctext, segs)
                    first_idx = idx
                if idx is not None and fb:
                    stats["critic_parsed"] = 1
                    self._tally(parsed=1)
                    # the multi-error critic always keeps the erroneous segment: otherwise the first k-1 errors
                    # would stay in the context while the k-th is removed
                    keep = segs[:idx] if (self.splice == "before" and not self.kerr) else segs[:idx + 1]
                    if self.cut_mode == "end" and not capped:
                        keep = segs
                        stats["cut_at_end"] = 1
                    if self.fb_mode == "generic":
                        fb = GENERIC_FEEDBACK
                    if self.fb_answer:
                        fb = f"{fb}\nThe correct final answer is {gt}."
                        stats["fb_answer"] = 1
                    kept_txt = "\n\n".join(keep)
                    _k = _tok_prefix_for_text(self.tokenizer, resp, kept_txt)
                    kept_ids = resp[:_k]
                    # the rewrite context must match the kept tokens exactly, so decode kept_ids
                    # instead of using kept_txt (the two can differ in whitespace)
                    kept_txt = self.tokenizer.decode(kept_ids, skip_special_tokens=True)
                    ctx = kept_txt + (FEEDBACK_TMPL.format(fb=fb) if self.use_feedback else "\n\n")
                    ctx_ids = self.tokenizer.encode(ctx, add_special_tokens=False)
                    budget = max(256, rlen - len(kept_ids))
                    if self.trunc:
                        budget = min(budget, self.trunc_max)
                    with simple_timer("repair", metrics):
                        r: TokenOutput = await self.server_manager.generate(
                            request_id=uuid4().hex, prompt_ids=prompt_ids + ctx_ids,
                            sampling_params={**sampling_params, "max_tokens": budget},
                            mm_processor_kwargs=mmk)
                    _rep_ids = list(r.token_ids)
                    _rep_lp = list(r.log_probs) if r.log_probs is not None else None
                    if self.trunc:
                        _kc = _cut_at_blank_line(self.tokenizer, _rep_ids)
                        _rep_ids = _rep_ids[:_kc]
                        if _rep_lp is not None:
                            _rep_lp = _rep_lp[:_kc]
                        stats["repair_tokens"] = _kc
                    # training sequence = kept prefix (mask 0) + student rewrite (mask 1); the feedback is not in it
                    _resp_r = (kept_ids + _rep_ids)[:rlen]
                    _npre = 0
                    if self.prefix_loss and first_idx:
                        _npre = min(_tok_prefix_for_text(self.tokenizer, resp, "\n\n".join(segs[:first_idx])), len(kept_ids))
                        stats["prefix_loss_tokens"] = _npre
                    _mask_r = ([1] * _npre + [0] * (len(kept_ids) - _npre) + [1] * len(_rep_ids))[:rlen]
                    # with KEEP_CORRECT an incorrect rewrite is discarded and the original rollout is used (num_turns=2)
                    _accept = True
                    if self.keep_correct:
                        _accept = await self._verify_async(
                            self.tokenizer.decode(_resp_r, skip_special_tokens=True), gt)
                        stats["repair_verified"] = int(_accept)
                        self._tally(**({"repair_ok": 1} if _accept else {"repair_bad": 1}))
                    if _accept:
                        resp, mask = _resp_r, _mask_r
                        stats["repaired"] = 1
                        type(self)._CNT["repaired"] += 1
                        metrics.setdefault("num_preempted", -1)
                        # rollout log-probs: the original rollout's on the prefix (_k counts original tokens,
                        # so they align), the rewrite's own on the rewrite
                        _lp = None
                        if out.log_probs is not None:
                            _rlp = _rep_lp if _rep_lp is not None else [0.0] * len(_rep_ids)
                            _lp = (list(out.log_probs[:_k]) + _rlp)[:len(resp)]
                            if len(_lp) < len(resp):
                                _lp += [0.0] * (len(resp) - len(_lp))
                        # L_fb: the teacher rescores the rewrite tokens with the feedback in its context. Only the
                        # inputs are passed on here; the teacher forward pass runs in
                        # AgentLoopWorker (agent_loop.py), which holds the teacher client. The number of rewrite
                        # tokens is taken after truncation to rlen.
                        _fb = {}
                        if os.environ.get("CRITIC_OPD_LOSS") == "fb":
                            _nrep = max(0, len(resp) - len(kept_ids))
                            if _nrep > 0:
                                _fb = {"critic_fb_prompt_ids": prompt_ids + ctx_ids,
                                       "critic_fb_n_kept": len(kept_ids),
                                       "critic_fb_n_repair": _nrep}
                        # num_turns=4 marks a repaired sample (used by the trainer for metrics and L_center)
                        return AgentLoopOutput(
                            prompt_ids=prompt_ids, response_ids=resp, response_mask=mask,
                            response_logprobs=_lp,
                            multi_modal_data=mm, mm_processor_kwargs=mmk, num_turns=4,
                            metrics=metrics, extra_fields={**(r.extra_fields or {}),
                                                           "critic_opd": stats,
                                                           **_fb})

        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = out.num_preempted if out.num_preempted is not None else -1
        return AgentLoopOutput(
            prompt_ids=prompt_ids, response_ids=resp, response_mask=[1] * len(resp),
            response_logprobs=out.log_probs[:rlen] if out.log_probs is not None else None,
            multi_modal_data=mm, mm_processor_kwargs=mmk, num_turns=2, metrics=metrics,
            extra_fields={**(out.extra_fields or {}), "critic_opd": stats})
