from __future__ import annotations
import os, re, json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from lxml import etree
from vllm import LLM, SamplingParams

# ------------------------------
# Config
# ------------------------------
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
MAX_NEW_TOKENS = 2500
SUMMARY_WORD_LIMIT = 50
INCLUDE_FEWSHOTS_DEFAULT = True
K_TARGET_DEFAULT = 1
N_NEIGH_DEFAULT = 20
# MAX_MODEL_CONTEXT_LEN = 32768
MAX_MODEL_CONTEXT_LEN = 25000

SYSTEM_ROLE = f"""You are an expert terminal session annotator. Your goal is to generate concise summaries of user actions.
Rules:
- summaries must not exceed {SUMMARY_WORD_LIMIT} words
- depth changes: -1=enter subtask, 0=continue same, +1=exit one level
- output must be valid JSON
""".strip()

FEWSHOTS_BLOCK = """
EXAMPLES (for format/logic only — do not output these)

NOTE: Event XML often shows keystroke-by-keystroke input with echoed characters, not full commands.

DEPTH SEMANTICS:
- depth = -1: STARTING a new subtask (entering deeper level)
- depth = 0:  CONTINUING at same level (ongoing work)
- depth = +1: FINISHING a subtask (returning to parent level)

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE A — Starting a new subtask (depth = -1)
Context: User opens a text editor from the command line
neighbor_tail:
  - id=10 depth=0  summary="List directory contents"
  - id=11 depth=0  summary="Change to project folder"
currDepth before target: 0

input xml:
<event>
  <user_input>n</user_input><system_output>n</system_output>
  <user_input>a</user_input><system_output>a</system_output>
  <user_input>n</user_input><system_output>n</system_output>
  <user_input>o</user_input><system_output>o</system_output>
  <user_input> </user_input><system_output> </system_output>
  <user_input>c</user_input><system_output>c</system_output>
  <user_input>o</user_input><system_output>o</system_output>
  <user_input>n</user_input><system_output>n</system_output>
  <user_input>f</user_input><system_output>f</system_output>
  <user_input>i</user_input><system_output>i</system_output>
  <user_input>g</user_input><system_output>g</system_output>
  <user_input>.</user_input><system_output>.</system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <user_input>x</user_input><system_output>x</system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <system_output>[nano editor opens]</system_output>
</event>

Expected output:
{"annotation": "Open config.txt in nano text editor.", "depth": -1}

Why depth = -1? User is STARTING a new subtask (text editor) - entering deeper level.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE B — Continuing at same level (depth = 0)
Context: User is working inside the editor, making edits
neighbor_tail:
  - id=20 depth=-1 summary="Open config.txt in nano text editor"
  - id=21 depth=0  summary="Navigate to database section"
currDepth before target: -1

input xml:
<event>
  <user_input>[Ctrl-W]</user_input>
  <system_output>Search: </system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <user_input>i</user_input><system_output>i</system_output>
  <user_input>m</user_input><system_output>m</system_output>
  <user_input>e</user_input><system_output>e</system_output>
  <user_input>o</user_input><system_output>o</system_output>
  <user_input>u</user_input><system_output>u</system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <system_output>[cursor moves to "timeout" line]</system_output>
</event>

Expected output:
{"annotation": "Search for timeout setting in config file.", "depth": 0}

Why depth = 0? User is CONTINUING work at the same level (still in editor) - ongoing task.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE C — Finishing a subtask (depth = +1)
Context: User saves and exits the editor
neighbor_tail:
  - id=30 depth=-1 summary="Open config.txt in nano editor"
  - id=31 depth=0  summary="Modify timeout value to 30"
  - id=32 depth=0  summary="Save changes to config file"
currDepth before target: -1

input xml:
<event>
  <user_input>[Ctrl-X]</user_input>
  <system_output>Save modified buffer? (Y/N)</system_output>
  <user_input>Y</user_input>
  <system_output>[exiting nano]</system_output>
  <system_output>user@laptop:~/project$ </system_output>
</event>

Expected output:
{"annotation": "Exit nano editor and return to shell.", "depth": 1}

Why depth = +1? User is FINISHING the editor subtask and returning to parent (shell) - exiting level.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE D — Starting a pager subtask (depth = -1)
Context: User views a log file with less
neighbor_tail:
  - id=40 depth=0  summary="Navigate to logs directory"
currDepth before target: 0

input xml:
<event>
  <user_input>l</user_input><system_output>l</system_output>
  <user_input>e</user_input><system_output>e</system_output>
  <user_input>s</user_input><system_output>s</system_output>
  <user_input>s</user_input><system_output>s</system_output>
  <user_input> </user_input><system_output> </system_output>
  <user_input>a</user_input><system_output>a</system_output>
  <user_input>p</user_input><system_output>p</system_output>
  <user_input>p</user_input><system_output>p</system_output>
  <user_input>.</user_input><system_output>.</system_output>
  <user_input>l</user_input><system_output>l</system_output>
  <user_input>o</user_input><system_output>o</system_output>
  <user_input>g</user_input><system_output>g</system_output>
  <system_output>[less pager opens showing log contents]</system_output>
</event>

Expected output:
{"annotation": "Open app.log file in less pager.", "depth": -1}

Why depth = -1? User is STARTING a new subtask (pager) - entering deeper level.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE E — Continuing within pager (depth = 0)
Context: User scrolls through the log file
neighbor_tail:
  - id=50 depth=-1 summary="Open app.log file in less pager"
  - id=51 depth=0  summary="Navigate to beginning of log"
currDepth before target: -1

input xml:
<event>
  <user_input>/</user_input>
  <system_output>/</system_output>
  <user_input>E</user_input><system_output>E</system_output>
  <user_input>R</user_input><system_output>R</system_output>
  <user_input>R</user_input><system_output>R</system_output>
  <user_input>O</user_input><system_output>O</system_output>
  <user_input>R</user_input><system_output>R</system_output>
  <system_output>[highlighting ERROR matches]</system_output>
</event>

Expected output:
{"annotation": "Search for ERROR keyword in log file.", "depth": 0}

Why depth = 0? User is CONTINUING work at the same level (still in pager) - ongoing task.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE F — Finishing pager subtask (depth = +1)
Context: User exits the pager
neighbor_tail:
  - id=60 depth=-1 summary="Open app.log in less pager"
  - id=61 depth=0  summary="Search for ERROR keyword in log"
  - id=62 depth=0  summary="Review error timestamps"
currDepth before target: -1

input xml:
<event>
  <user_input>q</user_input>
  <system_output>[exiting less]</system_output>
  <system_output>user@laptop:~/logs$ </system_output>
</event>

Expected output:
{"annotation": "Exit less pager and return to shell.", "depth": 1}

Why depth = +1? User is FINISHING the pager subtask and returning to parent (shell) - exiting level.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE G — Starting Python interpreter (depth = -1)
Context: User launches interactive Python session
neighbor_tail:
  - id=70 depth=0  summary="Check Python version installed"
currDepth before target: 0

input xml:
<event>
  <user_input>p</user_input><system_output>p</system_output>
  <user_input>y</user_input><system_output>y</system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <user_input>h</user_input><system_output>h</system_output>
  <user_input>o</user_input><system_output>o</system_output>
  <user_input>n</user_input><system_output>n</system_output>
  <user_input>3</user_input><system_output>3</system_output>
  <system_output>Python 3.10.4</system_output>
  <system_output>>>>></system_output>
</event>

Expected output:
{"annotation": "Launch Python3 interactive interpreter.", "depth": -1}

Why depth = -1? User is STARTING a new subtask (Python REPL) - entering deeper level.

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE H — Exiting Python interpreter (depth = +1)
Context: User exits Python back to shell
neighbor_tail:
  - id=80 depth=-1 summary="Launch Python3 interpreter"
  - id=81 depth=0  summary="Import json module and test parsing"
  - id=82 depth=0  summary="Print parsed dictionary contents"
currDepth before target: -1

input xml:
<event>
  <user_input>e</user_input><system_output>e</system_output>
  <user_input>x</user_input><system_output>x</system_output>
  <user_input>i</user_input><system_output>i</system_output>
  <user_input>t</user_input><system_output>t</system_output>
  <user_input>(</user_input><system_output>(</system_output>
  <user_input>)</user_input><system_output>)</system_output>
  <system_output>user@laptop:~$ </system_output>
</event>

Expected output:
{"annotation": "Exit Python interpreter and return to shell.", "depth": 1}

Why depth = +1? User is FINISHING the Python session and returning to parent (shell) - exiting level.

═══════════════════════════════════════════════════════════════════════════════

KEY PRINCIPLES:
1. Use depth = -1 when ENTERING a new tool/context (editor, pager, interpreter, debugger, etc.)
2. Use depth = 0 when CONTINUING work in the current context
3. Use depth = +1 when EXITING/CLOSING a tool/context back to parent
4. Think of depth as task nesting: -1 = push onto stack, 0 = work at current level, +1 = pop from stack
""".strip()

SYSTEM_ROLE = """You are an expert terminal session annotator. Your goal is to generate concise summaries of user actions.
Rules:
- summaries must not exceed {SUMMARY_WORD_LIMIT} words
- depth changes: -1=enter subtask, 0=continue same, +1=exit one level
- output must be valid JSON
""".strip()

@dataclass
class Event:
    idx: int
    xml: str
    depth_xml: Optional[int] = None
    summary_xml: Optional[str] = None

class Model1Annotator:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        summary_word_limit: int = SUMMARY_WORD_LIMIT,
        include_fewshots: bool = INCLUDE_FEWSHOTS_DEFAULT,
        n_neighbors: int = N_NEIGH_DEFAULT,
    ) -> None:
        self.model_id = model_id
        self.summary_word_limit = summary_word_limit
        self.include_fewshots = include_fewshots
        self.n_neighbors = n_neighbors

        self.events: List[Event] = []
        self.pred: Dict[int, Dict] = {}

        # Load vLLM once
        print(f"Loading model with vLLM: {self.model_id}")
        self.llm = LLM(
            model=self.model_id,
            gpu_memory_utilization=0.9,
            max_model_len=MAX_MODEL_CONTEXT_LEN,
            trust_remote_code=True,
            dtype="bfloat16",
        )
        self.tokenizer = self.llm.get_tokenizer()
        print("Model loaded.")

    # ---------- public API ----------

    def add_event(self, event_xml: str) -> int:
        """Append a new <event>...</event> string. Returns its index."""
        ev_xml = event_xml.strip()
        if not ev_xml.startswith("<event"):
            ev_xml = f"<event>{ev_xml}</event>"
        idx = len(self.events)
        self.events.append(Event(idx=idx, xml=ev_xml))
        return idx

    def annotate_latest(self) -> Tuple[int, Optional[int], Optional[str]]:
        """Annotate only the most recently added event."""
        if not self.events:
            raise ValueError("No events to annotate.")
        idx = len(self.events) - 1
        self._annotate_upto(idx, k_target=1)
        v = self.pred.get(idx, {})
        return idx, v.get("depth"), v.get("summary")

    def annotate_through(self, upto_idx: int, k_target: int = K_TARGET_DEFAULT) -> None:
        """Annotate events up to index `upto_idx` (inclusive)."""
        if upto_idx < 0 or upto_idx >= len(self.events):
            raise IndexError("upto_idx out of range.")
        self._annotate_upto(upto_idx, k_target=k_target)

    def get_event(self, idx: int) -> Event:
        return self.events[idx]

    def get_all_predictions(self) -> List[Tuple[int, Optional[int], Optional[str]]]:
        out = []
        for i, ev in enumerate(self.events):
            out.append((i, ev.depth_xml, ev.summary_xml))
        return out
  
    def format_summary_depth_lines(self) -> str:
      lines: List[str] = []
      for i in range(len(self.events)):
          ev = self.events[i]
          if ev.summary_xml is None or ev.depth_xml is None:
              continue  # only print annotated
          lines.append(ev.summary_xml.strip())
          lines.append(str(int(ev.depth_xml)))  # ensure int -> str

      return "\n".join(lines) + ("\n" if lines else "")

 ################### Internal Helpers ###################

    def _compute_curr_depth_upto(self, idx: int) -> int:
        curr = 0
        for i in range(idx):
            dep = self.events[i].depth_xml
            if dep is None:
                continue
            if dep == -1:
                curr -= 1
            elif dep > 0:
                curr += dep
        return curr

    def _make_flush_package(self, upto_idx: int, k_target: int) -> Dict:
        target_idxs = list(range(max(0, upto_idx - k_target + 1), upto_idx + 1))
        start_neigh = max(0, target_idxs[0] - self.n_neighbors)
        neighbor_idxs = list(range(start_neigh, target_idxs[0]))

        def _sum(i: int) -> str:
            if 0 <= i < len(self.events):
                s = self.events[i].summary_xml
                return s if s else "???"
            return "???"

        def _dep(i: int) -> int:
            if 0 <= i < len(self.events):
                d = self.events[i].depth_xml
                return d if d is not None else 999
            return 999

        neighbor_info = [f"- id={i} depth={_dep(i)}  summary={_sum(i)}" for i in neighbor_idxs]
        target_events = [self.events[i].xml for i in target_idxs if 0 <= i < len(self.events)]
        curr_depth = self._compute_curr_depth_upto(target_idxs[0])

        return {
            "target_idxs": target_idxs,
            "neighbor_info": neighbor_info,
            "target_events": target_events,
            "currDepth": curr_depth,
        }

    def _build_instruction(self, pkg: Dict) -> str:
        neighbor_items = []
        for line in pkg.get("neighbor_info", []):
            m = re.match(r"- id=(\d+) depth=(-?\d+)\s+summary=(.+)", line)
            if m:
                nid, ndepth, nsummary = m.groups()
                neighbor_items.append({"id": nid, "depth": ndepth, "summary": nsummary})

        neighbors_xml = "\n".join(
            [f'    <neighbor id="{n["id"]}" depth="{n["depth"]}">{n["summary"]}</neighbor>'
             for n in neighbor_items]
        ) or "    <neighbor>(none)</neighbor>"

        targets_xml = "\n".join(
            [f'  <target id="{idx}">\n{xml}\n  </target>'
             for idx, xml in zip(pkg["target_idxs"], pkg["target_events"])]
        )

        examples_xml = f"\n<examples>\n{FEWSHOTS_BLOCK}\n</examples>" if self.include_fewshots else ""

        extra = (
            "- do not copy xml tags or attributes; no repeated phrases\n"
            "- do not mention an address that was not explicitly mentioned in the event\n"
            "- if the target event contains an <annotation> tag or depth value ignore it\n"
            "- if there are no neighbors then the depth you output should be 0"
        )

        prompt = f"""<role>you are an event annotator for a linux terminal session.</role>

<output_format>
  {{"annotation": "<one sentence (≤ {self.summary_word_limit} words)>", "depth": <An integer greater than or equal to -1>}}
</output_format>

<think_first>
- Keep reasoning CONCISE and FOCUSED (2–3 sentences)
- In <think>...</think>: analyze the command, check depth logic, then conclude
- Use neighbors ONLY for continuity; do not invent context.
</think_first>

<rules>
- keystrokes appear separately; combine them to form the full command before interpreting it
- depth is an integer (≥ -1); -1 for subevent (new task), 0 same level, >0 exit levels
- maintain stack invariant: currDepth ≤ 0; if depth == -1 → currDepth -= 1; if depth > 0 → currDepth += depth
- write action-oriented summaries; avoid "user", "they", "typed", "inputs", "enters a command"
{extra}
</rules>{examples_xml}

<instruction>
for each target_event, output exactly one json with "annotation" first, then "depth".
</instruction>

<inputs>
  <curr_depth_max>{pkg.get("currDepth")}</curr_depth_max>
  <neighbors>
{neighbors_xml}
  </neighbors>
  <target_events>
{targets_xml}
  </target_events>
</inputs>"""
        return prompt

    def _build_messages(self, instruction: str) -> List[Dict[str, str]]:
        system = f"You are an expert terminal session annotator. Summaries ≤ {self.summary_word_limit} words; depth: -1/0/+k; output valid JSON."
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction},
        ]

    def _generate(self, messages: List[Dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        params = SamplingParams(
            temperature=0.0,
            max_tokens=MAX_NEW_TOKENS,
            repetition_penalty=1.2,
            skip_special_tokens=True,
        )
        outs = self.llm.generate([prompt], params)
        return outs[0].outputs[0].text.strip()

    def _parse_json_stream(self, text: str) -> List[Tuple[int, str]]:
        dec = json.JSONDecoder()
        i, n = 0, len(text)
        out: List[Tuple[int, str]] = []

        def add(obj):
            ann = obj.get("annotation")
            dep = obj.get("depth")
            if isinstance(ann, str):
                if isinstance(dep, str):
                    try:
                        dep = int(dep)
                    except Exception:
                        return
                if isinstance(dep, int) and dep >= -1:
                    out.append((dep, ann))

        while i < n:
            while i < n and text[i].isspace():
                i += 1
            if i >= n:
                break
            try:
                obj, end = dec.raw_decode(text, i)
            except json.JSONDecodeError:
                j = text.find("\n", i)
                if j == -1:
                    break
                i = j + 1
                continue
            i = end
            if isinstance(obj, dict):
                add(obj)
            elif isinstance(obj, list):
                for it in obj:
                    if isinstance(it, dict):
                        add(it)
        return out

    def _apply_pairs(self, pairs: List[Tuple[int, str]], target_idxs: List[int]) -> None:
        for (depth, summary), idx in zip(pairs, target_idxs):
            # Local invariant enforcement
            if depth < -1:
                depth = -1
            live_curr = self._compute_curr_depth_upto(idx)
            temp_curr = live_curr
            if depth == -1:
                temp_curr -= 1
            elif depth > 0:
                temp_curr += depth
            if temp_curr > 0:
                depth = 0

            self.pred[idx] = {"depth": depth, "summary": summary}
            self.events[idx].depth_xml = depth
            self.events[idx].summary_xml = summary

    def _annotate_upto(self, upto_idx: int, k_target: int) -> None:
        pkg = self._make_flush_package(upto_idx=upto_idx, k_target=k_target)
        instr = self._build_instruction(pkg)
        messages = self._build_messages(instr)

        full_out = self._generate(messages)
        # If the model emits <think>...</think>, strip to the JSON part
        json_part = full_out.split("</think>", 1)[1].strip() if "</think>" in full_out else full_out

        pairs = self._parse_json_stream(json_part)
        if not pairs:
            # fallback: no parse; mark depth 0 with a generic summary
            pairs = [(0, "No-op or unclear action.")]
        self._apply_pairs(pairs, pkg["target_idxs"])
