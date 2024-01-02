import os
import click
import torch
from tqdm import tqdm
from model import MyAwesomeModel
import matplotlib.pyplot as plt

from data import mnist


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    #Train model and plot training loss progress
    epochs = 10
    steps = 0
    train_losses = []

    for e in tqdm(range(epochs)):
        running_loss = 0
        for images, labels in train_set:
            steps += 1

            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_losses.append(running_loss / len(train_set))

    # Save plot of training loss in current folder
    plot_path = os.path.join(os.path.dirname(__file__), "training_loss.png")
    plt.plot(train_losses, label='Training loss')
    plt.legend(frameon=False)
    plt.savefig(plot_path)

    # Save model
    torch.save(model, os.path.join(os.path.dirname(__file__), "model.pt"))

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = mnist()

    accuracy = 0
    with torch.no_grad():
        for images, labels in tqdm(test_set, total=len(test_set)):
            log_ps = model(images)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
        
    print(f"Accuracy: {accuracy/len(test_set)}") 

cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
