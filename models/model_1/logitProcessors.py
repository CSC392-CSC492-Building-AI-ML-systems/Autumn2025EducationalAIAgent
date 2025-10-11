from transformers import LogitsProcessor
from typing import List, Dict, Set, Optional

class _Trie:
    def __init__(self, kids=None, term=False): self.k=kids or {}; self.t=term
    def insert(self, ids: List[int]):
        n=self
        for i in ids: n=n.k.setdefault(i,_Trie())
        n.t=True

def _build_trie(tok, texts: List[str]) -> _Trie:
    root=_Trie()
    for s in texts:
        ids=tok.encode(s, add_special_tokens=False)
        if ids: root.insert(ids)
    return root

class JsonAnnotationDepthProcessor(LogitsProcessor):
    """
    Forces: {"annotation":"...","depth": D} with D ∈ {-1,0,1..9}.
    Uses a token-trie to whitelist exact depth strings, covering composite tokens.
    """

    def __init__(self, tokenizer, allow_newlines_in_annotation: bool=False):
        import torch
        self.tok=tokenizer; self.torch=torch; self.allow_newlines=allow_newlines_in_annotation

        E=lambda s:self.tok.encode(s, add_special_tokens=False)
        self.pref_annot=E('{"annotation": "')
        self.pref_depth=E(', "depth": ')

        # Build trie over the allowed depth literals
        self.depth_trie=_build_trie(self.tok, ["-1","0","1","2","3","4","5","6","7","8","9"])

        # ids
        def one(ch): 
            ids=E(ch); return ids[0] if len(ids)==1 else None
        self.id_quote=one('"'); self.id_space=one(' '); self.id_rbrace=one('}')
        self.id_nl=one('\n')

        # state
        self.state="start"; self.i=0
        # trie decoding state for depth
        self.active_nodes=[self.depth_trie]  # list of current trie nodes (prefix candidates)
        self.depth_done=False

    def _force(self, scores, tok_id: Optional[int]):
        if tok_id is None: return scores
        m=self.torch.full_like(scores, float("-inf")); m[:,tok_id]=scores[:,tok_id]; return m

    def _mask_to(self, scores, allowed: Set[int]):
        m=self.torch.full_like(scores, float("-inf"))
        if allowed: m[:,list(allowed)]=scores[:,list(allowed)]
        return m

    def __call__(self, input_ids, scores):
        last = input_ids[0, -1].item()

        # 1) force '{"annotation": "'
        if self.state=="start":
            if self.i<len(self.pref_annot):
                scores=self._force(scores, self.pref_annot[self.i]); self.i+=1
                if self.i==len(self.pref_annot): self.state="annotation"
            return scores

        # 2) free annotation (optionally block newlines); detect closing quote
        if self.state=="annotation":
            if not self.allow_newlines and self.id_nl is not None:
                vocab=scores.shape[-1]; scores=self._mask_to(scores, set(range(vocab))-{self.id_nl})
            if last==self.id_quote:
                self.state="after_annot"; self.i=0
            return scores

        # 3) force ', "depth": '
        if self.state=="after_annot":
            if self.i<len(self.pref_depth):
                scores=self._force(scores, self.pref_depth[self.i]); self.i+=1
                if self.i==len(self.pref_depth): self.state="depth"
            return scores

        # 4) whitelist depth via trie: only {-1,0,1..9}
        if self.state=="depth":
            # compute next-allowed ids from all active trie nodes
            allowed:set[int]=set()
            for node in self.active_nodes:
                allowed.update(node.k.keys())
            # if a terminal was just matched, go to tail (allow only '}' or spaces)
            new_active=[]
            for node in self.active_nodes:
                if node.t: self.depth_done=True
            if self.depth_done:
                # only '}' (and optional spaces) after a complete depth literal
                tail=set()
                if self.id_rbrace is not None: tail.add(self.id_rbrace)
                if self.id_space  is not None: tail.add(self.id_space)
                return self._mask_to(scores, tail)

            # normal step: constrain to union of children
            scores=self._mask_to(scores, allowed)

            # advance trie state based on chosen last token
            next_nodes=[]
            for node in self.active_nodes:
                if last in node.k: next_nodes.append(node.k[last])
            # initialize active nodes if we're at the first depth token (no match yet)
            if not self.active_nodes or not next_nodes:
                # seed from root if starting now
                self.active_nodes=[]
                root=self.depth_trie
                if last in root.k: self.active_nodes=[root.k[last]]
            else:
                self.active_nodes=next_nodes
            return scores

        # 5) after '}' we’re done (no extra constraints)
        return scores
