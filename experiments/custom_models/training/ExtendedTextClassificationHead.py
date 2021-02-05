from farm.modeling.prediction_head import TextClassificationHead
import torch


class ExtendedTextClassificationHead(TextClassificationHead):
    def logits_to_probs(self, logits, return_class_probs, **kwargs):
        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(logits)
        if return_class_probs:
            probs = probs.cpu().numpy()
        else:
            pred_ids = logits.argmax(1)
            probs = torch.max(probs, dim=1)[0]
            probs = probs.cpu().numpy()
            probs = [val if pred_ids[i] == 1 else 1 - val for i, val in enumerate(probs)]
        return probs
