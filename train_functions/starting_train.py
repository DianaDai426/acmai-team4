import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pytorch_metric_learning import losses, miners
import os
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image

def starting_train(
    train_dataset, val_dataset, model, hyperparameters, n_eval, summary_path, device='cpu'
):
    writer = SummaryWriter()
    model = model.to(device)
    
    
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
        summary_path:    Path where Tensorboard summaries are located.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = losses.TripletMarginLoss()
    miner = miners.BatchEasyHardMiner(pos_strategy='all', neg_strategy='hard')


    data = pd.read_csv("/content/train.csv")
    data = data.sample(frac=1, random_state=1) # Shuffle data
    # train_data = data.iloc[:int(0.9*len(data))]
    # test_data = data.iloc[int(0.9*len(data)) + 1:]
    train_data = data.iloc[0:200]
    test_data = data.iloc[200:300]

    train_eval_dataset = EvaluationDataset(
        train_data,
        "/content/acmai-team4/corners.csv",
        "/content/train",
        train=True,
        drop_duplicate_whales=True, # If you set this to True, your evaluation accuracy will be lower!!
                                    # If you set this to False, evaluate() will take longer!!
                                    # Recommendation: set this to True during training, and when you're done,
                                    # create a new dataset with drop_duplicate_whales=False to get a final
                                    # evaluation metric.
    )
    train_eval_dataset.to(device)
    
    test_eval_dataset = EvaluationDataset(
        test_data,
        "/content/acmai-team4/corners.csv",
        "/content/train",
        train=False,
        drop_duplicate_whales=False,
    )
    
    test_eval_dataset.to(device)

    train_eval_loader = torch.utils.data.DataLoader(train_eval_dataset, batch_size=64)
    test_eval_loader = torch.utils.data.DataLoader(test_eval_dataset, batch_size=64)


    # Initialize summary writer (for logging)
    if summary_path is not None:
        writer = torch.utils.tensorboard.SummaryWriter(summary_path)

    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        trainLosses = []
        model.train()
        # Loop over each batch in the dataset
        for i, batch in enumerate(train_loader):

            batch_inputs = batch[1][0]
            batch_labels = batch[0][0]
            # print("inputs")
            # print(len(batch_inputs))
            # print(batch_inputs)
            # print("labels")
            # print(batch_labels)
            # print(batch[0])
            for i in range(1, 4): # set to batch_size
                    batch_inputs = torch.cat((batch_inputs, batch[1][i]), 0)
                    batch_labels = torch.cat((batch_labels, batch[0][i]), 0)

            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            # print(batch_labels)
            print(f"\rIteration {i + 1} of {len(train_loader)} ...", end="")


            #batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)

            # main body of your training
            optimizer.zero_grad()
            embeddings = model(batch_inputs) # images is a batch of images
            # print(embeddings)
            hard_triplets = miner(embeddings, batch_labels)
            #batch_outputs = model(batch_inputs)
           # print(f"batch size:\n{batch_outputs.shape()}\n\n")
            loss = loss_fn(embeddings, batch_labels, hard_triplets)
            loss.backward()
            trainLosses.append(loss)
            optimizer.step()
        print('End of epoch loss:', round((sum(trainLosses)/len(train_dataset)).item(), 3))


        # Periodically evaluate our model + log to Tensorboard
        #TODO: implement new evaluate code! https://gist.github.com/franktzheng/706fde69a652488a389455678438d4f0
        if step % n_eval == 0:
            # print(len(train_eval_loader))
            accuracy = evaluate(train_eval_loader, test_eval_loader, model)
            print(f"Accuracy: {accuracy}")
            writer.add_scalar(f"Accuracy", accuracy, epoch)

        step += 1

    print()


class EvaluationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        crop_info_path='/content/acmai-team4/corners.csv',
        image_folder='/content/train.csv',
        train=True,
        drop_duplicate_whales=True,
    ):
        self.data = data
        self.crop_info = pd.read_csv(crop_info_path, index_col="Image")
        self.image_folder = image_folder

        self.device = None

        if train:
            self.data = self.data[self.data.Id != "new_whale"]
        if drop_duplicate_whales:
            self.data = self.data.drop_duplicates(subset="Id")

    def to(self, device):
        self.device = device
        return self

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_file, whale_id = row.Image, row.Id

        """
        You may want to modify the code STARTING HERE...
        """
        bbox = self.crop_info.loc[row.Image]
        image = Image.open(os.path.join(self.image_folder, image_file))
        image = image.convert('RGB') # Maybe change this // done, changed to RGB
        image = image.crop((bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]))

        preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize((0.5,), (0.5,)), # and maybe this too
            ]
        )
        image = preprocess(image)
        # print(image)
        """
        ... and ENDING HERE. In particular, we converted the image to grayscale with a
        size of 224x448. You probably want to change that. //DONE
        """

        image = image.to(self.device)

        return image, whale_id

    def __len__(self):
        return len(self.data)


def evaluate(train_loader, test_loader, model, final=False):
    """
    Evaluates model performance. Both `train_loader` and `test_loader` should be
    instances of `EvaluationDataset`.
    """

    model.eval()

    """
    STEP 1. Compute TRAIN SET embeddings! We will use these embeddings to
    compare test images to.
    """

    # train_whale_ids[i] is the whale id corresponding to train_embeddings[i]
    train_embeddings = []
    train_whale_ids = []
    with torch.no_grad():  # Parentheses are important :)
        for batch in train_loader:
            images, whale_ids = batch
            # print(images[0].shape)
            batch_embeddings = model.forward(images)

            train_embeddings += list(batch_embeddings)
            train_whale_ids += list(whale_ids)

    # This will convert a list to a tensor
    train_embeddings = torch.stack(train_embeddings)

    """
    STEP 2. Compute TEST SET embeddings!
    """

    test_embeddings = []
    test_whale_ids = []
    with torch.no_grad():
        for batch in test_loader:
            images, whale_ids = batch
            batch_embeddings = model.forward(images)

            test_embeddings += list(batch_embeddings)
            test_whale_ids += list(whale_ids)

    """
    STEP 3. Compute the model's ACCURACY!
    """
    if final:
        accuracy = compute_final_accuracy(
            train_embeddings, train_whale_ids, test_embeddings, test_whale_ids
        )
    else:
        accuracy = compute_accuracy(
            train_embeddings, train_whale_ids, test_embeddings, test_whale_ids
        )

    model.train()

    return accuracy


def compute_final_accuracy(train_embeddings, train_ids, test_embeddings, test_ids):
    """
    Same as compute_accuracy, but will identify the optimal threshold for predicting
    "new_whale".
    """

    """
    NOTE: You may have to modify some of the values below depending on your choice of
    triplet loss margin (and other hyperparameters). Currently, the thresholds being
    tried are between 0.2 and 0.65.
    """

    """
    NOTE: This is actually bad practice because we are using the "test" set to determine
    what the threshold for predicting "new_whale" is. It is actually best to divide the
    dataset into train/validation/test sets, and use the validation set for these
    purposes, only touching the test set to get a final performance metric. But in
    practice, the threshold found would be similar even if there was a validation set.
    """

    best_threshold, best_accuracy = None, 0
    for i in range(10):
        # Threshold will range from 0.2 to 0.65
        threshold = 0.2 + 0.05 * i
        accuracy = compute_accuracy(
            train_embeddings, train_ids, test_embeddings, test_ids, threshold
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    print(f"Best threshold: {best_threshold}, Best accuracy: {best_accuracy}")
    return best_accuracy


def compute_accuracy(
    train_embeddings, train_ids, test_embeddings, test_ids, threshold=1000
):
    """
    Computes test accuracy. If the distance between a test embedding and every train
    embedding is at least `threshold`, then "new_whale" will be predicted.
    """

    correct, total = 0, 0

    for whale_id, embedding in zip(test_ids, test_embeddings):
        # This line will compute the distance between the test embedding and EVERY train
        # embedding
        distances = torch.norm(train_embeddings - embedding.view((1, 64)), dim=1)

        min_index = torch.argmin(distances)
        prediction = (
            "new_whale" if distances[min_index] > threshold else train_ids[min_index]
        )
        if prediction == whale_id:
            correct += 1
        total += 1

    return correct / total