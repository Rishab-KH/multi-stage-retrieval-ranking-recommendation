import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTowerModel(nn.Module):
    """
    Two-tower (bi-encoder) model with:
    - User tower: user embedding → MLP → L2-normalized output
    - Item tower: concat(item_emb, aisle_emb, dept_emb) → MLP → L2-normalized output

    Content embeddings (aisle, department) allow the item tower to generalize across
    products sharing catalog structure, addressing cold-start and improving retrieval quality.
    MLP towers enable non-linear feature interactions vs. pure lookup tables.
    """

    AISLE_EMB_DIM = 32
    DEPT_EMB_DIM = 16

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_aisles: int,
        num_depts: int,
        emb_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        # --- User tower ---
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.user_tower = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, emb_dim),
        )

        # --- Item tower ---
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.aisle_emb = nn.Embedding(num_aisles, self.AISLE_EMB_DIM)
        self.dept_emb = nn.Embedding(num_depts, self.DEPT_EMB_DIM)

        item_input_dim = emb_dim + self.AISLE_EMB_DIM + self.DEPT_EMB_DIM
        self.item_tower = nn.Sequential(
            nn.Linear(item_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, emb_dim),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.normal_(self.aisle_emb.weight, std=0.01)
        nn.init.normal_(self.dept_emb.weight, std=0.01)
        for m in list(self.user_tower) + list(self.item_tower):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward_user(self, user_idx: torch.LongTensor) -> torch.Tensor:
        x = self.user_emb(user_idx)
        x = self.user_tower(x)
        return F.normalize(x, p=2, dim=-1)

    def forward_item(
        self,
        item_idx: torch.LongTensor,
        aisle_idx: torch.LongTensor,
        dept_idx: torch.LongTensor,
    ) -> torch.Tensor:
        x = torch.cat([
            self.item_emb(item_idx),
            self.aisle_emb(aisle_idx),
            self.dept_emb(dept_idx),
        ], dim=-1)
        x = self.item_tower(x)
        return F.normalize(x, p=2, dim=-1)

    def get_user_embedding(self, user_idx: torch.LongTensor) -> torch.Tensor:
        return self.forward_user(user_idx)

    def get_item_embedding(
        self,
        item_idx: torch.LongTensor,
        aisle_idx: torch.LongTensor,
        dept_idx: torch.LongTensor,
    ) -> torch.Tensor:
        return self.forward_item(item_idx, aisle_idx, dept_idx)

    def get_all_item_embeddings(
        self,
        item_aisle: torch.LongTensor,
        item_dept: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Return normalized embeddings for all items. Shape: (num_items, emb_dim).
        Always runs on the model's device and returns a CPU tensor.
        item_aisle / item_dept may be on any device; they are moved internally.
        """
        device = next(self.parameters()).device
        item_idx = torch.arange(self.item_emb.num_embeddings, device=device)
        return self.forward_item(
            item_idx,
            item_aisle.to(device),
            item_dept.to(device),
        ).detach().cpu()
