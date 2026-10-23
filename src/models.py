import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        pad_idx: int,
        dropout: float = 0.0,
        positional_encoding: str = "none",
        max_len: int = 64,
    ) -> None:
        super().__init__()

        self.positional_encoding = build_positional_encoding(
            max_len=max_len,
            embedding_dim=embedding_dim,
            kind=positional_encoding,
        )
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_idx
        )
        self.cell = nn.GRU(
            hidden_size=hidden_size,
            input_size=embedding_dim,
            num_layers=1,
            batch_first=True,
        )
        self.token_head = nn.Linear(
            in_features=hidden_size, out_features=vocab_size, bias=True
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding(x)

        if self.positional_encoding is not None:
            embedding = self.positional_encoding(embedding)

        output, hidden = self.cell(embedding)

        output = self.dropout(output)

        token_logits = self.token_head(output)

        return token_logits

    def step(
        self,
        token_ids: torch.Tensor,
        hidden: torch.Tensor | None = None,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.embedding(token_ids)

        if self.positional_encoding is not None:
            embedding = self.positional_encoding(embedding, offset=offset)

        output, new_hidden = self.cell(embedding, hidden)

        logits = self.token_head(output)
        logits = logits.squeeze(1)

        return logits, new_hidden

    @torch.no_grad()
    def generate(
        self,
        token_to_id: dict[str, int],
        id_to_token: dict[int, str],
        prefix: str = "",
        temperature: float = 1.0,
        max_new_tokens: int = 20,
    ) -> str:
        if temperature <= 0 or max_new_tokens <= 0:
            raise ValueError("temperature and max_new_tokens must be > 0")

        prefix = prefix.lower()
        prefix_id = [token_to_id["<SOS>"]]
        for token in prefix:
            prefix_id.append(token_to_id[token])

        self.eval()

        device = self.embedding.weight.device

        hidden = None

        for position, token_id in enumerate(prefix_id):
            token = torch.tensor([[token_id]], dtype=torch.long, device=device)
            logits, hidden = self.step(token, hidden, offset=position)

        result = prefix

        for i in range(max_new_tokens):
            next_logits = logits[0].clone()
            next_logits[token_to_id["<SOS>"]] = -float("inf")
            next_logits[token_to_id["<PAD>"]] = -float("inf")

            probs = torch.softmax(next_logits / temperature, dim=0)
            next_id = torch.multinomial(probs, num_samples=1).item()

            if next_id == token_to_id["<EOS>"]:
                break

            result += id_to_token[next_id]

            token = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits, hidden = self.step(token, hidden, offset=len(prefix_id) + i)

        return result

    @torch.no_grad()
    def encode_sequence(self, names: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        embedding = self.embedding(names)

        if self.positional_encoding is not None:
            embedding = self.positional_encoding(embedding)

        output, hidden = self.cell(embedding)

        mask = names != self.pad_idx
        eos_positions = mask.sum(dim=1) - 1
        batch_indices = torch.arange(names.shape[0], device=names.device)
        last_hidden = output[batch_indices, eos_positions, :]

        return output, last_hidden.unsqueeze(0)

    def encode(self, names: torch.Tensor) -> torch.Tensor:
        output, last_hidden = self.encode_sequence(names)

        return last_hidden


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        pad_idx: int,
        dropout: float = 0.0,
        positional_encoding: str = "none",
        max_len: int = 64,
    ) -> None:
        super().__init__()

        self.positional_encoding = build_positional_encoding(
            kind=positional_encoding,
            embedding_dim=embedding_dim,
            max_len=max_len,
        )
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_idx
        )
        self.cell = nn.GRU(
            hidden_size=hidden_size,
            input_size=embedding_dim,
            num_layers=1,
            batch_first=True,
        )
        self.token_head = nn.Linear(
            in_features=hidden_size, out_features=vocab_size, bias=True
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        hidden: torch.Tensor,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        embedding = self.embedding(tokens)

        if self.positional_encoding is not None:
            embedding = self.positional_encoding(embedding, offset=offset)

        output, hidden_new = self.cell(embedding, hidden)
        output = self.dropout(output)
        logits = self.token_head(output)

        return logits, hidden_new


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bridge = nn.Linear(
            in_features=encoder.cell.hidden_size,
            out_features=decoder.cell.hidden_size,
            bias=True,
        )
        self.encoder.requires_grad_(False)

    def forward(self, source: torch.Tensor, target_input: torch.Tensor) -> torch.Tensor:
        encoder_hidden = self.encoder.encode(source)
        hidden = self.bridge(encoder_hidden)
        logits, hidden_new = self.decoder(target_input, hidden)

        return logits

    @torch.no_grad()
    def translate(
        self,
        name: str,
        latin_to_id: dict[str, int],
        cyrillic_to_id: dict[str, int],
        id_to_cyrillic: dict[int, str],
        max_new_tokens: int = 20,
    ) -> str:

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")

        self.eval()

        name = name.lower()
        name_id = [latin_to_id["<SOS>"]]
        for token in name:
            name_id.append(latin_to_id[token])
        name_id.append(latin_to_id["<EOS>"])

        device = self.bridge.weight.device

        name_id = torch.tensor([name_id], dtype=torch.long, device=device)

        hidden = self.encoder.encode(name_id)
        hidden_for_decoder = self.bridge(hidden)

        start_token = torch.tensor(
            [[cyrillic_to_id["<SOS>"]]], dtype=torch.long, device=device
        )
        logits, hidden_new = self.decoder(start_token, hidden_for_decoder)

        result = []

        for i in range(max_new_tokens):
            next_logits = logits[0, 0, :].clone()
            next_logits[cyrillic_to_id["<SOS>"]] = -float("inf")
            next_logits[cyrillic_to_id["<PAD>"]] = -float("inf")

            next_id = next_logits.argmax().item()
            if next_id == cyrillic_to_id["<EOS>"]:
                break

            result.append(id_to_cyrillic[next_id])

            token = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits, hidden_new = self.decoder(token, hidden_new, offset=i + 1)

        return "".join(result)


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        attention_dim: int,
    ) -> None:
        super().__init__()
        self.attention_dim = attention_dim

        self.query_projection = nn.Linear(
            in_features=query_dim, out_features=attention_dim, bias=False
        )
        self.key_projection = nn.Linear(
            in_features=context_dim, out_features=attention_dim, bias=False
        )
        self.value_projection = nn.Linear(
            in_features=context_dim, out_features=attention_dim, bias=False
        )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        source_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Q = self.query_projection(query)
        K = self.key_projection(context)
        V = self.value_projection(context)

        scores = Q @ K.transpose(-2, -1) / self.attention_dim**0.5
        mask = (~source_mask).unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))

        if attention_mask is not None:
            scores = scores.masked_fill(
                ~attention_mask,
                float("-inf"),
            )

        weights = torch.softmax(scores, dim=-1)
        attended_context = weights @ V

        return attended_context, weights.unsqueeze(1)


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        pad_idx: int,
        encoder_hidden_size: int,
        attention_dim: int,
        dropout: float = 0.0,
        positional_encoding: str = "none",
        max_len: int = 64,
        n_heads: int = 1,
    ) -> None:
        super().__init__()

        self.positional_encoding = build_positional_encoding(
            kind=positional_encoding,
            embedding_dim=embedding_dim,
            max_len=max_len,
        )
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_idx
        )
        if n_heads == 1:
            self.attention = Attention(
                query_dim=embedding_dim,
                context_dim=encoder_hidden_size,
                attention_dim=attention_dim,
            )
        else:
            self.attention = MultiHeadAttention(
                n_heads=n_heads,
                query_dim=embedding_dim,
                context_dim=encoder_hidden_size,
                attention_dim=attention_dim,
            )
        self.cell = nn.GRU(
            hidden_size=hidden_size,
            input_size=embedding_dim + attention_dim,
            num_layers=1,
            batch_first=True,
        )
        self.token_head = nn.Linear(
            in_features=hidden_size, out_features=vocab_size, bias=True
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        hidden: torch.Tensor,
        encoder_output: torch.Tensor,
        source_mask: torch.Tensor,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        embedding = self.embedding(tokens)

        if self.positional_encoding is not None:
            embedding = self.positional_encoding(embedding, offset=offset)

        attended_context, weights = self.attention(
            query=embedding, context=encoder_output, source_mask=source_mask
        )
        decoder_input = torch.cat([embedding, attended_context], dim=-1)
        output, hidden_new = self.cell(decoder_input, hidden)
        output = self.dropout(output)
        logits = self.token_head(output)

        return logits, hidden_new, weights


class AttentionSeq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: AttentionDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bridge = nn.Linear(
            in_features=encoder.cell.hidden_size,
            out_features=decoder.cell.hidden_size,
            bias=True,
        )
        self.encoder.requires_grad_(False)

    def forward(self, source: torch.Tensor, target_input: torch.Tensor) -> torch.Tensor:
        encoder_output, last_hidden = self.encoder.encode_sequence(source)
        source_mask = source != self.encoder.pad_idx
        hidden = self.bridge(last_hidden)
        logits, _, _ = self.decoder(target_input, hidden, encoder_output, source_mask)

        return logits

    @torch.no_grad()
    def translate(
        self,
        name: str,
        latin_to_id: dict[str, int],
        cyrillic_to_id: dict[str, int],
        id_to_cyrillic: dict[int, str],
        max_new_tokens: int = 20,
    ) -> str:

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")

        self.eval()

        name = name.lower()
        name_id = [latin_to_id["<SOS>"]]
        for token in name:
            name_id.append(latin_to_id[token])
        name_id.append(latin_to_id["<EOS>"])

        device = self.bridge.weight.device

        name_id = torch.tensor([name_id], dtype=torch.long, device=device)

        encoder_output, hidden = self.encoder.encode_sequence(name_id)
        hidden_for_decoder = self.bridge(hidden)

        start_token = torch.tensor(
            [[cyrillic_to_id["<SOS>"]]], dtype=torch.long, device=device
        )
        source_mask = name_id != self.encoder.pad_idx
        logits, hidden_new, weights = self.decoder(
            start_token, hidden_for_decoder, encoder_output, source_mask
        )

        result = []

        for i in range(max_new_tokens):
            next_logits = logits[0, 0, :].clone()
            next_logits[cyrillic_to_id["<SOS>"]] = -float("inf")
            next_logits[cyrillic_to_id["<PAD>"]] = -float("inf")

            next_id = next_logits.argmax().item()
            if next_id == cyrillic_to_id["<EOS>"]:
                break

            result.append(id_to_cyrillic[next_id])

            token = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits, hidden_new, weights = self.decoder(
                token, hidden_new, encoder_output, source_mask, offset=i + 1
            )

        return "".join(result)

    @torch.no_grad()
    def get_attention(
        self,
        name: str,
        translation: str,
        latin_to_id: dict[str, int],
        cyrillic_to_id: dict[str, int],
    ) -> tuple[torch.Tensor, list[str], list[str]]:
        previous_mode = self.training
        self.eval()

        try:
            device = self.bridge.weight.device

            source_tokens = ["<SOS>"] + list(name.lower()) + ["<EOS>"]
            query_tokens = ["<SOS>"] + list(translation.lower())

            source = torch.tensor(
                [[latin_to_id[token] for token in source_tokens]],
                dtype=torch.long,
                device=device,
            )

            target_input = torch.tensor(
                [[cyrillic_to_id[token] for token in query_tokens]],
                dtype=torch.long,
                device=device,
            )

            encoder_output, last_hidden = self.encoder.encode_sequence(source)
            hidden = self.bridge(last_hidden)
            source_mask = source != self.encoder.pad_idx

            _, _, weights = self.decoder(
                target_input,
                hidden,
                encoder_output,
                source_mask,
            )

            return weights[0], source_tokens, query_tokens

        finally:
            self.train(previous_mode)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        max_len: int,
    ) -> None:
        super().__init__()

        self.max_len = max_len
        self.embedding_dim = embedding_dim

        pe = torch.zeros(max_len, embedding_dim, dtype=torch.float32)

        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)

        even_dims = torch.arange(0, embedding_dim, 2, dtype=torch.float32)
        denominators = 10000.0 ** (even_dims / embedding_dim)
        angles = positions / denominators

        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles[:, : embedding_dim // 2])

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        seq_len = x.shape[1]
        if offset < 0 or offset + seq_len > self.max_len:
            raise ValueError("offset beyond the boundary")

        positions = self.pe[offset : offset + seq_len]

        return x + positions


class LearnedPositionalEncoding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        max_len: int,
    ) -> None:
        super().__init__()

        self.max_len = max_len
        self.position_embedding = nn.Embedding(
            num_embeddings=max_len, embedding_dim=embedding_dim
        )

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        seq_len = x.shape[1]
        if offset < 0 or offset + seq_len > self.max_len:
            raise ValueError("offset beyond the boundary")

        position_ids = torch.arange(
            offset, offset + seq_len, dtype=torch.long, device=x.device
        )

        positions = self.position_embedding(position_ids)

        return x + positions


def build_positional_encoding(
    kind: str,
    embedding_dim: int,
    max_len: int,
) -> nn.Module | None:
    if kind == "none":
        positional_encoding = None
    elif kind == "sinusoidal":
        positional_encoding = SinusoidalPositionalEncoding(
            embedding_dim=embedding_dim, max_len=max_len
        )
    elif kind == "learned":
        positional_encoding = LearnedPositionalEncoding(
            embedding_dim=embedding_dim, max_len=max_len
        )
    else:
        raise ValueError("positional_encoding must be none, sinusoidal or learned")

    return positional_encoding


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        query_dim: int,
        context_dim: int,
        attention_dim: int,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()

        if n_heads <= 0:
            raise ValueError("n_heads must be > 0")

        if output_dim is None:
            output_dim = attention_dim

        self.heads = nn.ModuleList()

        for _ in range(n_heads):
            self.heads.append(
                Attention(
                    query_dim=query_dim,
                    context_dim=context_dim,
                    attention_dim=attention_dim,
                )
            )

        self.projection = nn.Linear(
            in_features=n_heads * attention_dim, out_features=output_dim, bias=False
        )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        source_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended_context = []
        weights = []

        for head in self.heads:
            context_of_head, weights_of_head = head(
                query=query,
                context=context,
                source_mask=source_mask,
                attention_mask=attention_mask,
            )
            attended_context.append(context_of_head)
            weights.append(weights_of_head)

        attended_context = torch.cat(attended_context, dim=-1)
        attended_context = self.projection(attended_context)

        weights = torch.cat(weights, dim=1)

        return attended_context, weights


class FeedForward(nn.Module):
    def __init__(self, model_dim: int, feedforward_dim: int) -> None:
        super().__init__()

        self.linear1 = nn.Linear(
            in_features=model_dim, out_features=feedforward_dim, bias=True
        )
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(
            in_features=feedforward_dim, out_features=model_dim, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        n_heads: int,
        feedforward_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if n_heads <= 0 or model_dim % n_heads != 0:
            raise ValueError(
                "'n_heads' must be > 0 and 'model_dim % n_heads' must be 0"
            )

        self.self_attention = MultiHeadAttention(
            n_heads=n_heads,
            query_dim=model_dim,
            context_dim=model_dim,
            attention_dim=model_dim // n_heads,
            output_dim=model_dim,
        )

        self.feedforward = FeedForward(
            model_dim=model_dim, feedforward_dim=feedforward_dim
        )
        self.norm1 = nn.LayerNorm(normalized_shape=model_dim)
        self.norm2 = nn.LayerNorm(normalized_shape=model_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:

        self_context, self_weights = self.self_attention(
            query=x,
            context=x,
            source_mask=source_mask,
        )
        x = self.norm1(x + self.dropout(self_context))
        feedforward_output = self.feedforward(x)
        x = self.norm2(x + self.dropout(feedforward_output))

        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        n_heads: int,
        feedforward_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if n_heads <= 0 or model_dim % n_heads != 0:
            raise ValueError(
                "'n_heads' must be > 0 and 'model_dim % n_heads' must be 0"
            )

        self.self_attention = MultiHeadAttention(
            n_heads=n_heads,
            query_dim=model_dim,
            context_dim=model_dim,
            attention_dim=model_dim // n_heads,
            output_dim=model_dim,
        )

        self.cross_attention = MultiHeadAttention(
            n_heads=n_heads,
            query_dim=model_dim,
            context_dim=model_dim,
            attention_dim=model_dim // n_heads,
            output_dim=model_dim,
        )

        self.feedforward = FeedForward(
            model_dim=model_dim, feedforward_dim=feedforward_dim
        )
        self.norm1 = nn.LayerNorm(normalized_shape=model_dim)
        self.norm2 = nn.LayerNorm(normalized_shape=model_dim)
        self.norm3 = nn.LayerNorm(normalized_shape=model_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        self_context, self_weights = self.self_attention(
            query=x,
            context=x,
            source_mask=target_mask,
            attention_mask=causal_mask,
        )
        x = self.norm1(x + self.dropout(self_context))

        cross_context, cross_weights = self.cross_attention(
            query=x,
            context=encoder_output,
            source_mask=source_mask,
        )
        x = self.norm2(x + self.dropout(cross_context))

        feedforward_output = self.feedforward(x)
        x = self.norm3(x + self.dropout(feedforward_output))

        return x, cross_weights


class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        source_pad_idx: int,
        target_pad_idx: int,
        model_dim: int,
        n_heads: int,
        feedforward_dim: int,
        n_encoder_layers: int = 2,
        n_decoder_layers: int = 2,
        dropout: float = 0.0,
        positional_encoding: str = 'sinusoidal',
        max_len: int = 64,
    ) -> None:
        super().__init__()

        if n_encoder_layers < 1 or n_decoder_layers < 1:
            raise ValueError("Encoder and decoder must have at least one layer")

        self.source_pad_idx = source_pad_idx
        self.target_pad_idx = target_pad_idx
        self.model_dim = model_dim

        self.source_embedding = nn.Embedding(num_embeddings=source_vocab_size, padding_idx=source_pad_idx, embedding_dim=model_dim)
        self.target_embedding = nn.Embedding(num_embeddings=target_vocab_size, padding_idx=target_pad_idx, embedding_dim=model_dim)

        self.source_positional_encoding = build_positional_encoding(
            kind=positional_encoding,
            embedding_dim=model_dim,
            max_len=max_len,
        )
        self.target_positional_encoding = build_positional_encoding(
            kind=positional_encoding,
            embedding_dim=model_dim,
            max_len=max_len,
        )

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        
        for _ in range(n_encoder_layers):
            self.encoder_layers.append(
                TransformerEncoderLayer(
                    model_dim=model_dim,
                    n_heads=n_heads,
                    feedforward_dim=feedforward_dim,
                    dropout=dropout,
                )
            )

        for _ in range(n_decoder_layers):
            self.decoder_layers.append(
                TransformerDecoderLayer(
                    model_dim=model_dim,
                    n_heads=n_heads,
                    feedforward_dim=feedforward_dim,
                    dropout=dropout,
                )
            )

        self.token_head = nn.Linear(in_features=model_dim, out_features=target_vocab_size, bias=True)
        self.dropout = nn.Dropout(p=dropout)

    def encode(
        self,
        source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_mask = source != self.source_pad_idx
        x = self.source_embedding(source)

        if self.source_positional_encoding is not None:
            x = self.source_positional_encoding(x)

        x = self.dropout(x)

        for encoder in self.encoder_layers:
            x = encoder(x, source_mask)

        return x, source_mask

    def decode(
        self,
        target_input: torch.Tensor,
        encoder_output: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_mask = target_input != self.target_pad_idx

        seq_len = target_input.shape[1]
        causal_mask = torch.ones(
            (seq_len, seq_len),
            dtype=torch.bool,
            device=target_input.device
        ).tril(diagonal=0)

        x = self.target_embedding(target_input)

        if self.target_positional_encoding is not None:
            x = self.target_positional_encoding(x)

        x = self.dropout(x)

        for decoder in self.decoder_layers:
            x, cross_weights = decoder(
                x=x,
                encoder_output=encoder_output,
                source_mask=source_mask,
                target_mask=target_mask,
                causal_mask=causal_mask,
            )

        logits = self.token_head(x)

        return logits, cross_weights

    def forward(
        self,
        source: torch.Tensor,
        target_input: torch.Tensor,
    ) -> torch.Tensor:
        
        encoder_output, source_mask = self.encode(source)

        logits, cross_weights = self.decode(
            target_input=target_input,
            encoder_output=encoder_output,
            source_mask=source_mask
        )

        return logits

    @torch.no_grad()
    def translate(
        self,
        name: str,
        latin_to_id: dict[str, int],
        cyrillic_to_id: dict[str, int],
        id_to_cyrillic: dict[int, str],
        max_new_tokens: int = 20,
    ) -> str:

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")

        self.eval()

        name = name.lower()
        name_id = [latin_to_id["<SOS>"]]
        for token in name:
            name_id.append(latin_to_id[token])
        name_id.append(latin_to_id["<EOS>"])

        device = self.source_embedding.weight.device

        name_id = torch.tensor([name_id], dtype=torch.long, device=device)

        encoder_output, source_mask = self.encode(name_id)

        target_input = torch.tensor(
            [[cyrillic_to_id["<SOS>"]]], dtype=torch.long, device=device
        )

        result = []

        for i in range(max_new_tokens):
            logits, _ = self.decode(
                target_input=target_input,
                encoder_output=encoder_output,
                source_mask=source_mask,
            )
            next_logits = logits[0, -1, :].clone()

            next_logits[cyrillic_to_id["<SOS>"]] = -float("inf")
            next_logits[cyrillic_to_id["<PAD>"]] = -float("inf")

            next_id = next_logits.argmax().item()
            if next_id == cyrillic_to_id["<EOS>"]:
                break

            result.append(id_to_cyrillic[next_id])

            token = torch.tensor([[next_id]], dtype=torch.long, device=device)
            target_input = torch.cat([target_input, token], dim=1)

        return "".join(result)

    @torch.no_grad()
    def get_attention(
        self,
        name: str,
        translation: str,
        latin_to_id: dict[str, int],
        cyrillic_to_id: dict[str, int],
    ) -> tuple[torch.Tensor, list[str], list[str]]:
        previous_mode = self.training
        self.eval()

        try:
            device = self.source_embedding.weight.device

            source_tokens = ["<SOS>"] + list(name.lower()) + ["<EOS>"]
            query_tokens = ["<SOS>"] + list(translation.lower())

            source = torch.tensor(
                [[latin_to_id[token] for token in source_tokens]],
                dtype=torch.long,
                device=device,
            )

            target_input = torch.tensor(
                [[cyrillic_to_id[token] for token in query_tokens]],
                dtype=torch.long,
                device=device,
            )

            encoder_output, source_mask = self.encode(source)

            _, weights = self.decode(
                target_input=target_input,
                encoder_output=encoder_output,
                source_mask=source_mask,
            )

            return weights[0], source_tokens, query_tokens

        finally:
            self.train(previous_mode)