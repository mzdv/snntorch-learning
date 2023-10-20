import torch
import snntorch as snn
from matplotlib import pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from snntorch import spikegen

from torch import nn
import snntorch.spikeplot as splt

CUT_TRAINING_SHORT = False
TRAINING_CUTOFF = 1

dtype = torch.float32
device = 'cpu'

scaler = MinMaxScaler()

iris = datasets.load_iris()
# X = scaler.fit_transform(iris.data)
X = iris.data
y = iris.target

X_tensored = torch.tensor(X, dtype=torch.float32)

### Spikes
# X_spiked = spikegen.rate(X_tensored, num_steps=1)
# X_spiked = spikegen.rate_conv(X_tensored)
X_spiked = spikegen.rate_interpolate(X_tensored, num_steps=1)

# X_spiked = spikegen.latency(X_tensored, num_steps=1)
# X_spiked = spikegen.delta(X_tensored, threshold=1)
### End spikes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)

input_nodes = 4  # Sepal length, sepal width, petal length, petal width
hidden_neurons = 500  # Arbitrary number of hidden layer neurons
output_neurons = 3  # Setosa, Virginica, Versicolor

dt_steps = 500  # simulation steps
beta = 0.95  # used for LIF decay rate

epochs = 300


class SpikingIrisClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        #   What about changing the architecture to skip the transfer1 layer?
        #   We need something to map 500 to 3, they maybe have 3 neurons all the
        #   time in the transfer layer?

        #   In the end, it's neuromorphic, not neuroreal
        self.input = nn.Linear(input_nodes, hidden_neurons)
        self.lif1 = snn.Leaky(beta=beta)
        self.transfer1 = nn.Linear(hidden_neurons, output_neurons)
        self.output = snn.Leaky(beta=beta)

    def forward(self, x):
        # We need to init the membranes since they handle the spiking

        membrane_lif1 = self.lif1.init_leaky()
        membrane_output = self.output.init_leaky()

        spikes_output = []
        membrane_output_voltages = []

        input_current = self.input(x)
        spikes_lif1, membrane_lif1 = self.lif1(input_current, membrane_lif1)
        # because there is only one membrane, we just energized it

        transfer_current = self.transfer1(spikes_lif1)
        spikes_transfer, membrane_output = self.output(transfer_current,
                                                       membrane_output)

        spikes_output.append(spikes_transfer)
        membrane_output_voltages.append(membrane_output)

        return spikes_transfer, membrane_output


#   Pushing the whole dataset in every pass since it can cause overfitting when
#   we start minibatching and giving bad classifications
def train_classifier(classifier, loss_function, optimizer, inputs, actuals):
    print('Training started')
    print('\n')

    loss_history = []

    for epoch in range(epochs):
        print(f'Epoch: {epoch + 1}/{epochs}')
        classifier.train()

        spikes, membrane_voltages = classifier(
            torch.tensor(inputs).to(device, dtype=dtype))

        loss_values = torch.zeros(1, dtype=dtype, device=device)
        loss_values += loss_function(membrane_voltages,
                                     torch.LongTensor(actuals))

        optimizer.zero_grad()
        loss_values.backward()
        optimizer.step()

        current_loss_value = loss_values.item()
        loss_history.append(current_loss_value)
        print(f'Current loss: {current_loss_value}')

        # if epoch % 25 == 0:
        #     fig = plt.figure(facecolor="w", figsize=(10, 5))
        #     ax = fig.add_subplot(111)
        #     #  s: size of scatter points; c: color of scatter points
        #     splt.raster(spikes, ax, s=1.5, c="black")
        #     plt.title("Input Layer")
        #     plt.xlabel("Time step")
        #     plt.ylabel("Neuron Number")
        #     plt.show()

        if CUT_TRAINING_SHORT and current_loss_value < TRAINING_CUTOFF:
            break

    return loss_history


def test_classifier(classifier, inputs, actuals):
    print('Testing started')
    print('\n')

    with torch.no_grad():
        classifier.eval()

        input = torch.tensor(inputs).to(device, dtype=dtype)
        testing_spikes, _ = classifier(input)

        _, predicted = torch.max(testing_spikes, 1)
        tensored_actuals = torch.LongTensor(actuals)

        return (predicted == tensored_actuals).sum().item(), \
        tensored_actuals.size()[0]


classifier = SpikingIrisClassifier().to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-4,
                             betas=(0.9, 0.999))

training_loss_history = train_classifier(classifier, loss_function, optimizer,
                                         X_train, y_train)
correct_guesses, total_guesses = test_classifier(classifier, X_test, y_test)

# fig = plt.figure(facecolor="w", figsize=(10, 5))
# plt.plot(training_loss_history)
# plt.title("Loss Curve")
# plt.legend(["Train Loss"])
# plt.xlabel("Iteration")
# plt.ylabel("Loss")1
# plt.show()

print("------")
print(f"Testing state: {correct_guesses}/{total_guesses}")
print(f"Test Set Accuracy: {100 * correct_guesses / total_guesses:.2f}%")
print("------")
