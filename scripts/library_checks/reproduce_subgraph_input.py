import torch, timm
from torch import nn
from nn_lib.models.graph_module_plus import GraphModulePlus as G

class SingleInput(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
    def forward(self, x):
        return self.inner(x)

m = timm.create_model("vit_base_patch16_224", pretrained=False).eval()
gm = G.new_from_trace(SingleInput(m).eval()).squash_all_conv_batchnorm_pairs().eval()

down = G.new_from_copy(gm).extract_subgraph(inputs=["add_14"])
print([n.name for n in down.graph.nodes if n.op == "placeholder"])