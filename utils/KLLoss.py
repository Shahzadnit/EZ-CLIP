
import torch.nn.functional as F
import torch.nn as nn
import torch

class KLLoss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        print('=========using KL Loss=and has temperature and * bz==========')
        self.error_metric = error_metric

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss
    
    
class Motion_loss(nn.Module):
    
    def __init__(self):
        super(Motion_loss, self).__init__()

    def forward(self, x):
        delta = 1
        shifted_left = torch.cat([x[:, :1], x[:, :-1]], dim=1)
        shifted_right = torch.cat([x[:, 1:], x[:, -1:]], dim=1)

        # Compute central difference derivative
        central_diff = torch.abs(shifted_right - shifted_left) / 2.0
        C = torch.mean(torch.mean(torch.mean(central_diff, 1),1))
        b = torch.var(x, dim=1, keepdim=False)
        V = torch.mean(torch.mean(b, 1))
        L = C+V
        loss = 1/(L+delta)
        return loss
 
 
 
    
class Triplet_loss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self):
        super(Triplet_loss, self).__init__()
        self.margin = 2.0
    def cosine_distance(self,embeddings1, embeddings2):
        # Calculate cosine similarity
        similarity = 1 - F.cosine_similarity(embeddings1, embeddings2)
        return similarity
    
    def align_loss(self, x, y, alpha=2):
        return (self.margin - (x - y).norm(p=2, dim=1).pow(alpha).mean())


    def uniform_loss(self,x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def forward(self, triplet_samples):
        anchor_images = triplet_samples['anchor_image']
        anchor_texts = triplet_samples['anchor_text']
        positive_images = triplet_samples['positive_image']
        positive_texts = triplet_samples['positive_text']
        negative_images = triplet_samples['negative_image']
        negative_texts = triplet_samples['negative_text']

        # image_distance = F.pairwise_distance(anchor_images, positive_images) - F.pairwise_distance(anchor_images, negative_images) + self.margin
        # text_distance = F.pairwise_distance(anchor_texts, positive_texts) - F.pairwise_distance(anchor_texts, negative_texts) + self.margin
        
        # image_distance = self.cosine_distance(anchor_images, positive_images) - self.cosine_distance(anchor_images, negative_images) + self.margin
        # text_distance = self.cosine_distance(anchor_texts, positive_texts) - self.cosine_distance(anchor_texts, negative_texts) + self.margin
        # image_text_distance = self.cosine_distance(anchor_images, positive_texts) + self.cosine_distance(anchor_texts, positive_images) + self.cosine_distance(anchor_texts, negative_images) - self.cosine_distance(anchor_images, negative_texts) + self.margin

        # image_loss = torch.clamp(image_distance, min=0.0).mean()
        # text_loss = torch.clamp(text_distance, min=0.0).mean()
        # image_text_distance = torch.clamp(image_text_distance, min=0.0).mean()
        
        anchor_images = anchor_images/torch.norm(anchor_images, dim=-1,keepdim=True)
        positive_images = anchor_images/torch.norm(positive_images, dim=-1,keepdim=True)
        
        anchor_texts = anchor_texts/torch.norm(anchor_texts, dim=-1,keepdim=True)
        positive_texts = anchor_texts/torch.norm(positive_texts, dim=-1,keepdim=True)
        
        negative_images = negative_images/torch.norm(negative_images, dim=-1,keepdim=True)
        negative_texts = negative_texts/torch.norm(negative_texts, dim=-1,keepdim=True)
        
        image_alig_loss = self.align_loss( anchor_images, negative_images)
        text_alig_loss = self.align_loss( anchor_texts, negative_images)
        image_text_alig_loss = self.align_loss( anchor_texts, negative_texts)
        text_images_alig_loss = self.align_loss( anchor_images, negative_texts)
        # image_uniform_loss = self.uniform_loss(anchor_images)
        # text_uniform_loss = self.uniform_loss(anchor_texts)
        loss = (image_alig_loss + text_alig_loss + image_text_alig_loss + text_images_alig_loss)/4 #+ image_uniform_loss + text_uniform_loss
        # return (image_loss + text_loss + image_text_distance)/3
        return loss

    
