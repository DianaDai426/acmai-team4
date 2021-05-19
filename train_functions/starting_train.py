import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

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
    loss_fn = nn.CrossEntropyLoss()

    # Initialize summary writer (for logging)
    if summary_path is not None:
        writer = torch.utils.tensorboard.SummaryWriter(summary_path)

    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        losses = []
        model.train()
        # Loop over each batch in the dataset
        for i, batch in enumerate(train_loader):
            if i == 5:
                break
            batch_inputs = batch[1]
            batch_labels = batch[0]
            print(batch_labels)
            print(f"\rIteration {i + 1} of {len(train_loader)} ...", end="")


            #batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)

            # main body of your training
            optimizer.zero_grad()
            batch_outputs = model(batch_inputs)
            print(f"batch size:\n{batch_outputs.size()}\n\n")
            loss = loss_fn(batch_outputs, batch_labels)
            loss.backward()
            losses.append(loss)
            optimizer.step()
        print('End of epoch loss:', round((sum(losses)/len(train_dataset)).item(), 3))



        # Periodically evaluate our model + log to Tensorboard
        if step % n_eval == 0:
            # TODO:
            # Compute training loss and accuracy.
            # Log the results to Tensorboard.

            # TODO:
            # Compute validation loss and accuracy.
            # Log the results to Tensorboard.
            # Don't forget to turn off gradient calculations!
            
            train_loss, train_acc = evaluate(train_loader, model, loss_fn)
            val_loss, val_acc = evaluate(val_loader, model, loss_fn)
            
            writer.add_scalar(f"Training loss", train_loss, epoch)
            writer.add_scalar(f"Training Accuracy", train_acc, epoch)
            writer.add_scalar(f"Validation loss", val_loss, epoch)
            writer.add_scalar(f"Validation Accuracy", val_acc, epoch)

        step += 1

    print()


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """

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