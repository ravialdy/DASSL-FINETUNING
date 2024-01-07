from my_dassl.engine import TRAINER_REGISTRY, TrainerX
from my_dassl.optim import build_optimizer, build_lr_scheduler
from my_dassl.metrics import compute_accuracy
import torch
from tqdm import tqdm
import time, datetime
from torch import nn
import torch.nn.functional as F
from torchvision import models

@TRAINER_REGISTRY.register()
class FTVGG16(TrainerX):
    def build_model(self):
        self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.avgpool = nn.AdaptiveAvgPool2d(output_size=(2, 2))
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=10)
        )

        self.model = self.model.to(self.device)
        self.optim = build_optimizer(self.model, self.cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, self.cfg.OPTIM)
        self.register_model("vgg16", self.model, self.optim, self.sched)

        self.best_result = 0.0

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        output = self.model(image)
        loss = F.cross_entropy(output, label)
        self.model_backward_and_update(loss)

        acc = compute_accuracy(output, label)[0].item()
        loss_summary = {
            "loss": loss.item(),
            "acc": acc,
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
    
    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    def model_inference(self, input):
        return self.model(input)
    
    def after_train(self):
        print("Finish training")
        # all_last_acc = self.test()
        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")
        self.close_writer()
    
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
    
    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")  # Get current validation result
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result  # Update the best result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    model_name="model-best.pth.tar"
                )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)