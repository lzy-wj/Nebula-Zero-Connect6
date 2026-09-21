"""Raw-board TensorRT wrappers shared by inference-first candidates."""

import torch
from torch import nn
from torch.nn import functional as F


class FusedPairSelfPlayWrapper(nn.Module):
    """Fuse board encoding, rules and native low-rank pair outputs."""

    def __init__(self, model, compute_dtype=torch.float16, pair_rank=16):
        super().__init__()
        self.model = model
        self.compute_dtype = compute_dtype
        self.pair_rank = int(pair_rank)
        feature_dim = int(model.feature_dim)

        self.candidate_projection = nn.Linear(
            feature_dim,
            self.pair_rank,
            bias=False,
        )
        self.first_projection = nn.Linear(
            feature_dim,
            self.pair_rank,
            bias=False,
        )
        pair_value_hidden = max(64, feature_dim // 2)
        self.pair_global_projection = nn.Linear(
            feature_dim,
            pair_value_hidden,
        )
        self.pair_action_projection = nn.Linear(
            feature_dim,
            pair_value_hidden,
            bias=False,
        )
        self.pair_value_output = nn.Linear(pair_value_hidden, 1)

        kernels = torch.zeros((4, 1, 6, 6), dtype=torch.float32)
        kernels[0, 0, 0, :] = 1
        kernels[1, 0, :, 0] = 1
        kernels[2, 0] = torch.eye(6)
        kernels[3, 0] = torch.rot90(torch.eye(6), 1, [0, 1])
        self.register_buffer("threat_kernels", kernels)

    def _detect_threats(self, stones, blockers):
        padded_stones = F.pad(stones, (5, 5, 5, 5), value=0)
        counts = F.conv2d(padded_stones, self.threat_kernels)
        padded_blockers = F.pad(blockers, (5, 5, 5, 5), value=1)
        blocked = F.conv2d(padded_blockers, self.threat_kernels)
        counts = counts * blocked.eq(0).float()
        return (
            counts.ge(6).float().flatten(1).amax(1),
            counts.eq(5).float().flatten(1).amax(1),
            counts.eq(4).float().flatten(1).amax(1),
        )

    def forward(self, board):
        board = torch.where(board == 2, -torch.ones_like(board), board)
        stone_count = board.ne(0).sum(dim=(1, 2))
        rank = (stone_count + 1) // 2
        current_player = torch.where(rank.remainder(2).eq(1), -1, 1)
        player_view = current_player.view(-1, 1, 1)

        self_stones = board.eq(player_view).float().unsqueeze(1)
        opponent_stones = board.eq(-player_view).float().unsqueeze(1)
        empty = board.eq(0).float().unsqueeze(1)
        black_plane = current_player.eq(1).float().view(
            -1, 1, 1, 1
        ).expand(-1, 1, 19, 19)
        second_stone = (
            stone_count.gt(0) & stone_count.remainder(2).eq(0)
        ).float().view(-1, 1, 1, 1).expand(-1, 1, 19, 19)
        inputs = torch.cat(
            (self_stones, opponent_stones, empty, black_plane, second_stone),
            dim=1,
        ).to(self.compute_dtype)

        policy_logits, _, raw_value, features = self.model(
            inputs,
            return_features=True,
        )
        raw_value = raw_value.flatten()
        normalized = F.layer_norm(features, (self.model.feature_dim,))
        candidate_factors = self.candidate_projection(normalized)
        first_factors = self.first_projection(normalized)

        global_features = normalized.mean(dim=1)
        pair_hidden = (
            self.pair_global_projection(global_features).unsqueeze(1)
            + self.pair_action_projection(normalized)
        )
        pair_residual = self.pair_value_output(F.gelu(pair_hidden)).squeeze(2)
        parent_logit = 0.5 * (
            torch.log1p(raw_value.float().clamp(-0.999, 0.999))
            - torch.log1p(-raw_value.float().clamp(-0.999, 0.999))
        )
        pair_value = torch.tanh(parent_logit.unsqueeze(1) + pair_residual.float())

        policy_logits = policy_logits.float().masked_fill(
            empty.flatten(1).eq(0),
            -10_000.0,
        )
        policy = torch.softmax(policy_logits, dim=1)
        value = raw_value.float()

        my_win, my_five, my_four = self._detect_threats(
            self_stones,
            opponent_stones,
        )
        opponent_win, opponent_five, opponent_four = self._detect_threats(
            opponent_stones,
            self_stones,
        )
        stones_remaining = torch.where(
            stone_count.gt(0) & stone_count.remainder(2).eq(1),
            2,
            1,
        )
        value = torch.where(opponent_four.bool(), value.clamp(max=-0.2), value)
        value = torch.where(opponent_five.bool(), value.clamp(max=-0.25), value)
        can_win = my_five.bool() | (my_four.bool() & stones_remaining.ge(2))
        value = torch.where(can_win, torch.ones_like(value), value)
        value = torch.where(opponent_win.bool(), -torch.ones_like(value), value)
        value = torch.where(my_win.bool(), torch.ones_like(value), value)

        return (
            policy,
            value.unsqueeze(1),
            candidate_factors,
            first_factors,
            pair_value.to(self.compute_dtype),
        )
