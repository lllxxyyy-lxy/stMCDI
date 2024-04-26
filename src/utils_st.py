import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle


def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=300,
    foldername="",
):

    torch.manual_seed(0)
    np.random.seed(0)
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p0 = int(0.25 * config["epochs"])
    p1 = int(0.5 * config["epochs"])
    p2 = int(0.75 * config["epochs"])
    p3 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p0, p1, p2, p3], gamma=0.1
    )
    # history = {'train_loss':[], 'val_rmse':[]}
    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                # The forward method returns loss.
                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step()

        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            print("Start validation")
            model.eval()
            avg_loss_valid = 0
            # some initial settings
            val_nsample = 15
            val_scaler = 1
            mse_total = 0
            mae_total = 0
            evalpoints_total = 0

            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        output = model.evaluate(valid_batch, val_nsample)

                        (
                            samples,
                            c_target,
                            eval_points,
                            observed_points,
                            observed_time,
                        ) = output
                        samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                        c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                        eval_points = eval_points.permute(0, 2, 1)
                        observed_points = observed_points.permute(0, 2, 1)

                        samples_median = samples.median(dim=1)
                        mse_current = (
                            ((samples_median.values - c_target) * eval_points) ** 2
                        ) * (val_scaler**2)
                        mae_current = (
                            torch.abs((samples_median.values - c_target) * eval_points)
                        ) * val_scaler

                        mse_total += torch.sum(mse_current, dim=0)
                        evalpoints_total += torch.sum(eval_points, dim=0)

                        it.set_postfix(
                            ordered_dict={
                                "rmse_total": torch.mean(
                                    torch.sqrt(torch.div(mse_total, evalpoints_total))
                                ).item(),
                                "batch_no": batch_no,
                            },
                            refresh=True,
                        )

    if foldername != "":
        torch.save(model.state_dict(), output_path)



def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):
    torch.manual_seed(3407)
    np.random.seed(3407)

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)


                samples_median = samples.median(dim=1)

                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)


                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler**2)
                # print(f"mse_current: {mse_current}")
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points)
                ) * scaler
                mse_total += torch.sum(mse_current, dim=0)
                mae_total += torch.sum(mae_current, dim=0)
                evalpoints_total += torch.sum(eval_points, dim=0)
                # print(f"mse_total: {mse_total}")
                # print(f"evalpoints_total: {evalpoints_total}")
                it.set_postfix(
                    ordered_dict={
                        "rmse_total": torch.mean(
                            torch.sqrt(torch.div(mse_total, evalpoints_total))
                        ).item(),
                        "mae_total": torch.mean(
                            torch.abs(torch.div(mae_total, evalpoints_total))
                        ).item(),
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            #Use folloing code for saving generated results.
            with open(
                foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0)
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_observed_point = torch.cat(all_observed_point, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump(
                    [
                        # shape: [len(test_dataset), nsample, L, K]]
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            with open(foldername + "/result_nsample" + str(nsample) + ".pk", "wb") as f:
                pickle.dump(
                    [
                        torch.mean(
                            torch.sqrt(torch.div(mse_total, evalpoints_total))
                        ).item(),
                    ],
                    f,
                )
            rmse = torch.mean(torch.sqrt(torch.div(mse_total, evalpoints_total)))

            print(
                "RMSE:",
                torch.mean(torch.sqrt(torch.div(mse_total, evalpoints_total))).item(),
            )
            print(
                "MAE:",
                torch.mean(torch.abs(torch.div(mae_total, evalpoints_total))).item(),
            )


