import whisper
import torch
import torch.nn.functional as F

class SpeculativeWhisper:
    def __init__(self, config):
        self.device = config.device
        self.draft = whisper.load_model(config.draft_model).to(self.device)
        self.final = whisper.load_model(config.final_model).to(self.device)
        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)
        self.k = config.k
        self.max_tokens = config.max_tokens
        self.mel_dim_tiny = config.mel_dim_tiny
        self.mel_dim_large = config.mel_dim_large
        self.beam_search = config.beam_search
        self.beam_size = getattr(config, "beam_size", 5)
        self.top_p = getattr(config, "top_p", None)

    def topp_sample(self, logits):
        if self.top_p is not None:
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum > self.top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            idx = torch.multinomial(sorted_probs, 1)
            return sorted_idx.gather(-1, idx).squeeze(-1)
        return torch.argmax(logits, dim=-1)

    def _beam_search(self, tmp, draft_encoder, active_indices):
        beams = [(tmp, torch.zeros(tmp.size(0), device=self.device))]
        for _ in range(self.k):
            new_beams = []
            for seq, score in beams:
                logits = self.draft.decoder(seq, draft_encoder[active_indices])[:, -1]
                logp = F.log_softmax(logits, dim=-1)
                topk_logp, topk_idx = torch.topk(logp, self.beam_size, dim=-1)
                for b in range(self.beam_size):
                    nt = topk_idx[:, b]
                    ns = torch.cat([seq, nt[:, None]], dim=1)
                    new_beams.append((ns, score + topk_logp[:, b]))
            beams = sorted(new_beams, key=lambda x: x[1].sum().item(), reverse=True)[:self.beam_size]
        best = beams[0][0]
        draft = []
        for i in range(self.k):
            draft.append(best[:, -(self.k - i)])
        return torch.stack(draft, dim=1), best

    def decode(self, draft_encoder, final_encoder, max_tokens=None):
        max_tokens = max_tokens or self.max_tokens
        batch = final_encoder.size(0)
        tokens = [torch.full((1,), self.tokenizer.sot, device=self.device, dtype=torch.long) for _ in range(batch)]
        done = torch.zeros(batch, dtype=torch.bool, device=self.device)

        for _ in range(max_tokens):
            active_indices = (~done).nonzero(as_tuple=True)[0]
            if len(active_indices) == 0:
                break

            tmp = torch.nn.utils.rnn.pad_sequence(
                [tokens[i] for i in active_indices],
                batch_first=True,
                padding_value=self.tokenizer.sot,
            )

            with torch.no_grad():
                if self.beam_search:
                    draft, tmp = self._beam_search(tmp, draft_encoder, active_indices)
                else:
                    draft_list = []
                    for _ in range(self.k):
                        logits = self.draft.decoder(tmp, draft_encoder[active_indices])[:, -1]
                        next_tok = self.topp_sample(logits)
                        draft_list.append(next_tok)
                        tmp = torch.cat([tmp, next_tok[:, None]], dim=1)
                    draft = torch.stack(draft_list, dim=1)

            verify = torch.cat([tmp[:, :-draft.size(1)], draft[:, :-1]], dim=1)

            with torch.no_grad():
                logits = self.final.decoder(verify, final_encoder[active_indices])
                logp = F.log_softmax(logits, dim=-1)

            for idx, seq_idx in enumerate(active_indices):
                accepted = 0
                base = tokens[seq_idx].size(0) - 1
                for i in range(draft.size(1)):
                    pred = torch.argmax(logp[idx, base + i], dim=-1)
                    if pred == draft[idx, i]:
                        accepted += 1
                    else:
                        break

                if accepted > 0:
                    tokens[seq_idx] = torch.cat([tokens[seq_idx], draft[idx, :accepted]], dim=0)

                if accepted < draft.size(1):
                    pos = tokens[seq_idx].size(0) - 1
                    fb = self.topp_sample(logp[idx, pos])
                    tokens[seq_idx] = torch.cat([tokens[seq_idx], fb.unsqueeze(0)], dim=0)

                done[seq_idx] = tokens[seq_idx][-1] == self.tokenizer.eot

        return tokens

    def transcribe(self, audio_files, max_tokens=None):
        max_tokens = max_tokens or self.max_tokens
        audios = []

        for p in audio_files:
            a = whisper.load_audio(p)
            a = whisper.pad_or_trim(a)
            audios.append(torch.from_numpy(a))

        audios = torch.stack(audios).to(self.device)
        mel_tiny = torch.stack([whisper.log_mel_spectrogram(a, self.mel_dim_tiny) for a in audios]).to(self.device)
        mel_large = torch.stack([whisper.log_mel_spectrogram(a, self.mel_dim_large) for a in audios]).to(self.device)

        with torch.no_grad():
            draft_encoder = self.draft.encoder(mel_tiny)
            final_encoder = self.final.encoder(mel_large)

        batch_tokens = self.decode(draft_encoder, final_encoder, max_tokens)

        return [self.tokenizer.decode(t.tolist()) for t in batch_tokens]
