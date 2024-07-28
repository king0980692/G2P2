import torch
from tqdm import tqdm, trange
from collections import ChainMap


class TripletTrainer():
    def __init__(self,
                 device,
                 model,
                 n_epoch,
                 loss_cls,
                 update_times,
                 optimizer,
                 optimizer_params,
                 lr_scheduler):
        """
        Notes:
            Triplets trainer using Pairwise Loss
        """
        self.device = device
        self.model = model
        self.n_epoch = n_epoch
        self.loss_cls = loss_cls
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.update_times = update_times
        self.lr_scheduler = lr_scheduler

    def inference(self, model=None, data_loader=None, observed_dict=None, user_map=None, rv_user_map=None, item_map=None, rv_item_map=None):
        model = model if model is not None else self.model
        model = model.to(self.device)
        model.eval()
        query_pred = {}
        with torch.no_grad():
            for id, x_dict in enumerate(tqdm(data_loader)):
                fea = {k: v.to(self.device) for k, v in x_dict.items()}
                try:
                    observed_items = observed_dict[fea['u'][0].item()]
                except:
                    observed_items = []

                scores = model(fea)
                topk = torch.topk(scores, 1000).indices
                query_pred[x_dict['u'][0].item()] = [
                    t for t in topk.tolist() if t not in observed_items][:1000]

        return query_pred

    def extract_embedding(self, fea_name_list):

        out_embs = []
        for fea_name in fea_name_list:
            emb = self.model.embedding.embed_dict[fea_name].weight.data
            out_embs.append(emb)
        return out_embs

    def fit(self, train_dl, val_dl=None):

        self.model = self.model.to(self.device)
        pbar = tqdm(train_dl, total=self.update_times,
                    bar_format='{l_bar}{bar}|[{elapsed}<{remaining}, {rate_fmt}]',
                    )
        idx = 0
        for id, x_dict in enumerate(pbar):
            batch_size = x_dict['u'].shape[0]
            features = {k: v.to(self.device) for k, v in x_dict.items()}
            user_fea = {k: v for k, v in features.items() if 'u' in k}

            pos_item_fea = {k: v for k, v in features.items()
                            if 'i' in k and 'neg' not in k}
            neg_item_fea = {k: v for k,
                            v in features.items() if 'i' in k and 'neg' in k}

            self.optimizer.zero_grad()
            pos_score = self.model(ChainMap(user_fea, pos_item_fea))
            neg_score = self.model(ChainMap(user_fea, neg_item_fea))

            # regularization ## need check
            # reg = 0.0001 * \
            # sum(p.pow(2.0).sum() for p in self.model.embedding.embed_dict.parameters())

            loss = self.loss_cls(pos_score, neg_score)
            loss.backward()
            self.optimizer.step()
            if self.lr_scheduler:
                self.lr_scheduler.step()

            # pbar.set_description(f"{self.lr_scheduler.get_lr()[0]:.6f}")

            idx += batch_size  # accumalted update sampe times

            pbar.update(batch_size)
            pbar.refresh()  # to show immediately the update

            if idx >= self.update_times:
                break

        x_dict.pop('neg-i')  # delete negative sample from x_dict for inference
        return self.model, x_dict
