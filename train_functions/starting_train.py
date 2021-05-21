import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pytorch_metric_learning import losses, miners

def starting_train(
    train_dataset, val_dataset, model, hyperparameters, n_eval, summary_path
):
    # writer = SummaryWriter()
    
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
            if i == 5:
                break

            batch_inputs = batch[1][0]
            batch_labels = batch[0][0]
            # print(batch[0])
            for i in range(1, 8):
                    batch_inputs = torch.cat((batch_inputs, batch[1][i]), 0)
                    batch_labels = torch.cat((batch_labels, batch[0][i]), 0)
            
            print(batch_labels)
            print(f"\rIteration {i + 1} of {len(train_loader)} ...", end="")


            #batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)

            # main body of your training
            optimizer.zero_grad()
            embeddings = model(batch_inputs) # images is a batch of images
            hard_triplets = miner(embeddings, batch_inputs)
            #batch_outputs = model(batch_inputs)
           # print(f"batch size:\n{batch_outputs.shape()}\n\n")
            loss = loss_fn(embeddings, labels, hard_triplets)
            loss.backward()
            trainLosses.append(loss)
            optimizer.step()
        print('End of epoch loss:', round((sum(trainLosses)/len(train_dataset)).item(), 3))

    
    #each item of dataset is a single image
    #now make it so each item is 4 different images but same ID
    #grab 8 things from a dataset --> total of 8*4 images
    #now batch has 8 groups of 4
    #every whale in that batch appears 4 times
    # when u return the 4 whales from the single item of the datset and grab the batch out of it,' it will be in a weird format:
    #the batch will be of length 8 but each item inside will have 4 images inside it so we need to stack: torch.cat
    #new whale can't be treated like a normal class bc they aren't all the same whale --> they're all unidentfiied
    # 1) augment data --> for each whale: do a  horizontal flip  to the image, etc so that we ca make 4 images out of each single image
    # that deals with all the whales that have very few images
    # 2) instead of doing 4 whales for every item, return a set of 3 whales and then tag on an extra new_whale 
    # so you have 3 images of the same whale, and then a new whale. we do this so we can use the new_whale images


        # Periodically evaluate our model + log to Tensorboard
        if step % n_eval == 0:
            
            train_loss, train_acc = evaluate(train_loader, model, loss_fn)
            val_loss, val_acc = evaluate(val_loader, model, loss_fn)
            
            writer.add_scalar(f"Training loss", train_loss, epoch)
            writer.add_scalar(f"Training Accuracy", train_acc, epoch)
            writer.add_scalar(f"Validation loss", val_loss, epoch)
            writer.add_scalar(f"Validation Accuracy", val_acc, epoch)

        step += 1

    print()


def compute_accuracy(outputs, labels):

    n_correct = (torch.round(outputs) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn):
    """
    Computes the loss and accuracy of a model on the validation dataset.

    TODO!
    """
    
     # Compute validation loss and accuracy.
    # Log the results to Tensorboard.
    # Don't forget to turn off gradient calculations!
    model.eval()
    num_correct = 0
    loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images = batch[1]
            labels = batch[0]
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).int().sum()
            total += len(predictions)
            loss += loss_fn(outputs, labels)
            # print(i)
            
        # for i, (labels, images) in enumerate(val_dataset):
        #     # images = batch[1]
        #     # labels = batch[0]
        #     outputs = model(images)
        #     predictions = torch.argmax(outputs, dim=1)
        #     correct += (predictions == labels)
        #     total += 1
        #     loss += loss_fn(outputs, labels)
        #     print(i)
    return loss, (correct/total)
    
# conv_net.eval() # sets the net to evaluation mode to save memory

# with torch.no_grad(): # tell it not to keep track of gradients for this portion
#   total = 0
#   correct = 0
#   for batch in test_loader:
#     images, labels = batch
#     images = images.to(device)
#     labels = labels.to(device)
#     outputs = conv_net(images)
#     predictions = torch.argmax(outputs, dim=1)
#     correct += (predictions == labels).int().sum()
#     total += len(predictions)
#   print(correct / total)


