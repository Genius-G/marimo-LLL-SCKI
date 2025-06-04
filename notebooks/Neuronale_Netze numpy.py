import marimo

__generated_with = "0.13.10"
app = marimo.App(width="full")


app._unparsable_cell(
    r"""
    mo.md(
        rf\"\"\"
    # Neuronale Netze

    In der letzten Einheit haben wir das Perzeptron kennen gelernt, das durch Fehler lernt und in bestimmten Szenarien Daten richtig klassifizieren kann. Der Klassifikationsalgorithmus des Perzeptrons stößt allerdings schnell an seine Grenzen. In dieser Einheit schauen wir uns an, wie wir das Perzeptron schrittweise verbessern können. Diese Verbesserungen führen uns zu neuronalen Netzen, die die rasante Entwicklung der KI der letzten Jahre entscheidend prägten.

    KI wird in den nächsten Jahren immer mehr Aufgaben übernehmen, die jetzt noch von Menschen ausgeführt werden. Gleichzeitig schafft KI auch neue Berufe und Perspektiven. Eine wichtige Herausforderung der Zukuft ist u.a. die Gestaltung einer sinnvollen Zusammenarbeit zwischen Mensch und KI. 

    {mo.image(src=\"public/resources/img/artificial-intelligence.jpg\", alt=\"Deep Neural Network\", style={\"width\":\"50%\"}, caption=\"Dieses Bild wurde übrigens von einer KI erzeugt.\")}

    Zum Einstieg in diese Einheit rufen wir uns den Aufbau des Perzeptrons in Erinnerung. Das Perzeptron besteht aus einer festen Anzahl Inputs (abhängig von den Dimensionen der Punkte, die als Datengrundlage dienen), Gewichten mit denen die Eingaben mulipliziert und zusammen mit dem Bias addiert werden und einer Aktivierungsfunktion. Diesen Aufbau bezeichnen wir im Folgenden als <b>Neuron</b>.

    {mo.image(src=\"public/resources/img/perzeptron.png\", alt=\"perzeptron\", style={\"width\": \"70%\"})}

    ## Aufbau neuronaler Netze

    Im Folgenden ändern wir das Perzeptron Schritt für Schritt ab, um dessen Defizite zu beheben.

    ### Mehr als zwei Klassen klassifizieren und Performance steigern

    Um die Performance unserer KI zu steigern, schalten wir mehrere Neuronen hinter- und nebeneinander. Die Ausgabe eines Neurons dient nun als Eingabe von nachfolgenden Neuronen. Sind Neuronen parallel in einer Ebene angeordnet, wird die Gesamtheit dieser Neuronen als <b>Layer</b> (bzw. Schicht) bezeichnet. Das gesamte Konstrukt mehreren Neuronenschichten bezeichnet man als <b>neuronales Netz</b>. Wenn es mehrere verdeckte Schichten gibt, bezeichnet man das Netz als <b>tiefes neuronales Netz</b> (deep neural network).

    {mo.image(src=\"public/resources/img/nn1.png\", alt=\"perzeptron\", style={\"width\": \"60%\"})}

    Um nicht nur zwei Klassen von Datenpunkten klassifizieren zu können, wird die Ausgabe durch mehrere Neuronen erweitert. Die Nummer des Neurons, das den größten Wert in der Ausgabeschicht ausgibt, ist auch die Ausgabe des gesamten neuronalen Netzes. Wenn es also fünf Ausgabeneuronen gibt und das mittlere den größten Wert hat, dann weist das neuronale Netz den Datenpunkt der Klasse 2 zu (Outputs 0 bis 5). Bisher sind die Ausgaben der Neuronen allerdings entweder 0 oder 1, so dass es oft zu einem Gleichstand kommen kann. Nicht nur deswegen sollten wir die bisherige Aktivierungsfunktion durch eine geeignetere ersetzen.

    ### Neue Aktivierungsfunktion

    Das Perzeptron kann nur dann Datenpunkte verschiedener Klassen voneinander trennen, wenn die Datenpunkte der unterschiedlichen Klassen durch eine Gerade getrennt werden können. Das wird u.a. durch die Treppenfunktion verursacht, die wir als Aktivierungsfunktion verwenden. Außerdem gehen durch die Weiterleitung von entweder 0 oder 1 viele Informationen verloren, weil es keine Werte dazwischen gibt. Die <b>Sigmoidfunktion</b> $sig$ oder die <b>ReLU-Funktion</b> $relu$ sind in vielen Fällen besser als Aktivierungsfunktionen der Neuronen geeignet. Für unsere neuronalen Netze werden wir hauptsächlich die ReLU-Funktion verwenden.

    $$ sig(x) = \dfrac{e^x}{e^x + 1} $$


    $$ relu(x) = \left\{
    \begin{array}{ll}
    0, & x \leq 0 \\
    x, & \, \textrm{sonst} \\
    \end{array}
    \right. $$

    {mo.image(src=\"public/resources/img/sigmoid_and_relu.png\", alt=\"Sigmoid and ReLU\", style={\"width\": \"50%\"})}

    ### Softmax

    Jetzt fehlt nur noch eine kleine Änderung, um ein herkömmliches neuronales Netz zu erhalten. Wie im vorletzten Abschnitt bereits umrissen, wird die Klassifikation des Datenpunkts jetzt nicht mehr durch eine 0- oder 1-Ausgabe des letzten Neurons ermittelt, sondern durch die Nummer des Neurons in der Ausgabeschicht, das die größte Ausgabe hat. Durch die neue ReLU-Aktivierungsfunktion erhalten wir in der letzten Ausgabeschicht nicht mehr 0- oder 1-Ausgaben, sondern Werte größer oder gleich 0. 
    Um als Ausgabe des neuronalen Netzes die Wahrscheinlichkeit zu erhalten, mit der ein Datenpunkt einer Klasse zugeordnet wird, wird eine am Ende eine zusätzliche Schicht mit einer speziellen Aktivierungsfunktion (Softmax-Funktion) eingefügt, deren Gewichte nicht trainiert werden.

    {mo.image(src=\"public/resources/img/nn2.png\", alt=\"perzeptron\", style={\"width\": \"80%\"})}

    Jetzt sind wir bereit unser erstes neuronales Netz in Code umzusetzen. Damit wir nicht alles selbst implementieren müssen, verwenden wir die Bibliothek <i>Numpy</i>.

    ## Numpy

    Numpy bietet eine sehr einfache Weise, neuronale Netze zu konstruieren. Gehe das folgende Codefeld durch und führe es aus, um mit den Funktionsaufrufen vertraut zu werden. Wir konstruieren dabei das obige neuronale Netze mit vier Eingabe- und drei Ausgabeneuronen.
    \"\"\"
    )
    """,
    name="_"
)


@app.cell
def _(FullyConnectedLayer, np, relu, softmax):
    class Net:
        def __init__(self, num_in, num_out):
            self.name_model = "Netzi"

            # Define layers that mimic the torch layers
            self.fc1 = FullyConnectedLayer(num_in, 5)      # first layer: num_in -> 5 
            self.fc2 = FullyConnectedLayer(5, 5)             # second layer: 5 ->5
            self.fc3 = FullyConnectedLayer(5, num_out, bias=False)  # third layer: 5 -> num_out, no bias

        def forward(self, x):
            # First layer + ReLU
            x = relu(self.fc1.forward(x))
            # Second layer + ReLU
            x = relu(self.fc2.forward(x))
            # Third layer (no ReLU here)
            x = self.fc3.forward(x)
            # Softmax on output layer
            x = softmax(x)
            return x

        def __str__(self):
            # Simple representation of the network layers
            rep = f"Model name: {self.name_model}\n"
            rep += "Layer 1 (fc1):\n" + str(self.fc1) + "\n"
            rep += "Layer 2 (fc2):\n" + str(self.fc2) + "\n"
            rep += "Layer 3 (fc3):\n" + str(self.fc3)
            return rep

    # Create an instance of the network with 4 input features and 3 output features.
    erstes_nn = Net(4, 3)
    print(f"Hallo mein Name ist {erstes_nn.name_model}!\n")

    print("Das ist mein Aufbau:\n")
    print(erstes_nn, "\n")

    print("Und das sind meine zufällig initialisiert/en Gewichte:")
    # Display weights and biases of each layer
    print("\nfc1 layer parameters:\n", "Weights:\n", erstes_nn.fc1.weight)
    if erstes_nn.fc1.use_bias:
        print("Bias:\n", erstes_nn.fc1.bias)

    print("\nfc2 layer parameters:\n", "Weights:\n", erstes_nn.fc2.weight)
    if erstes_nn.fc2.use_bias:
        print("Bias:\n", erstes_nn.fc2.bias)

    print("\nfc3 layer parameters:\n", "Weights:\n", erstes_nn.fc3.weight)
    if erstes_nn.fc3.use_bias:
        print("Bias:\n", erstes_nn.fc3.bias)

    # Test input vector
    test_eingabe = np.array([1.0, 2.5, -1, 0])

    # Make predictions using the forward method.
    ausgabe = erstes_nn.forward(test_eingabe)
    print("\nAusgabe (Ergebnis des Vorwärtsdurchlaufs):", ausgabe)
    return (erstes_nn,)


@app.cell
def _(mo):
    mo.md(r"# Aufgabe 1")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ____
    
        <img style="float: left;" src="public/resources/img/laptop_icon.png" width=50 height=50 /> <br><br>
    
        <i>Im letzten Codefeld wurden unser erstes neuronales Netz erzeugt. Lies die gesuchten Gewichte anhand der letzten Ausgabe ab und überprüfe deine Eingabe, indem du das Codefeld ausführst. Runde gegebenenfalls die Eingaben auf die vierte Nachkommastelle ab.</i>
        """
    )
    return


@app.cell
def _(erstes_nn):
    from public.resources.code.help_functions import pruefe_gewichte

    # Ersetze die Nullen durch die richtigen Werte.

    # Gewicht zwischen dem ersten Neuron der Eingabeschicht und dem ersten Neuron der ersten verdeckten Schicht
    gewicht1 = 0

    # Bias des letzen Neurons der ersten verdeckten Schicht
    gewicht2 = 0

    # Bias des zweiten Neurons der zweiten verdeckten Schicht
    gewicht3 = 0

    # Gewicht zwischen dem vierten Neuron der zweiten verdeckten Schicht 
    # und dem dritten Neuron der dritten verdeckten Schicht
    gewicht4 = 0

    print(pruefe_gewichte(erstes_nn, gewicht1, gewicht2, gewicht3, gewicht4))
    return


@app.cell
def _(mo):
    mo.md(r"# Aufgabe 2")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ____
    
        <img style="float: left;" src="public/resources/img/laptop_icon.png" width=50 height=50 /> <br><br>
    
        <i>Jetzt bist du bereit ein neuronales Netz eigenständig zu konstruieren. Implementiere das abgebildete neuronale Netz und gib das Ergebnis des durchpropagierten Datenpunkts an.</i>
    
        <figure>
          <img src="public/resources/img/nn3.png" alt="neuronales Netz" style="width:60%">
          <figcaption></figcaption>
        </figure>
        """
    )
    return


@app.cell
def _(np):
    datenpunkt = np.array([1.0, 2.0])

    # Füge hier deinen Code ein.
    return


@app.cell
def _(mo):
    mo.accordion({
        "Tipp 1": "Sieh dir nochmal die Definition von `class Net` an.",
        "Tipp 2": "Du brauchst um ein eigenes neuronales Netz zu implementieren eine eigene Klasse mit 2 Eingaben und 2 Ausgaben sowie 2 versteckten Schichten",
        "Tipp 3": "Tipp 3",
        "Lösung":mo.md(
        r"""
    ```python
    class Net_1:
            def __init__(self, num_in, num_out):
                # First fully connected layer maps from num_in to 2
                self.fc1 = FullyConnectedLayer(num_in, 2)
                # Second fully connected layer maps from 2 to 2
                self.fc2 = FullyConnectedLayer(2, 2)

            def forward(self, x):
                x = relu(self.fc1.forward(x))
                x = self.fc2.forward(x)
                x = softmax(x)
                return x
    ```
    """
    ),
    })
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Bis jetzt haben wir zwar neuronale Netze konstruiert, aber sie noch nicht trainieren lassen. Die vorhandenen Trainingsdaten müssen wir nutzen, um die Gewichte so anzupassen, dass das neuronale Netz auf den Testdaten (die wir nicht für das Training benutzen) gute Ergebnisse erzielt. Im nächsten Abschnitt schauen wir uns an, wie das funktioniert.
    
        ## Backpropagation
    
        Der Algorithmus, der die Gewichte der neuronalen Netze abändert und ein entscheidender Faktor am Erfolg von Deep-Learning-Algorithmen ist, ist der <b>Backpropagation-Algorithmus</b>. Der Backpropagation-Algorithmus ist ein Optimierungsalgorithmus, d.h. bei der Funktion, die den Fehler des neuronalen Netzes beschreibt, wird (in diesem Fall) nach dem Minimum gesucht, weil wir den Fehler so klein wie möglich halten möchten. 
    
        Die Suche nach dem Minimum können wir uns mit folgendem Bild veranschaulichen. Ein Weihnachtsmann sitzt in seinem E-Schlitten auf einem Hügel und möchte den Weg ins Tal finden. Leider kennt er den Weg dahin nicht. Zu allem Überfluss ist es auch schon dunkel und sogar etwas nebelig ist, sodass er nur zehn Meter weit sehen kann. Er kann aber um sich herum erkennen, in welche Richtung der Hügel am steilsten abfällt. (In diese Richtung zeigt übrigens auch die Ableitung der Funktion, die das Gelände beschreibt.) Er stellt sein E-Schlitten so ein, dass er eine bestimmte Distanz in die Richtung des steilsten Abstiegs fährt, anschließend stoppt, die Richtung des Abstiegs noch einmal neu bestimmt und in diese Richtung wieder eine bestimmte Distanz fährt. Wenn alles optimal verläuft, findet er auf diese Weise den Weg ins Tal.
    
        <figure>
          <img src="public/resources/img/loss_function.png" alt="Verlustfunktion" style="width:60%">
          <figcaption></figcaption>
        </figure> 
    
        Analog dazu funktioniert auch die Optimierung bei neuronalen Netzen. Die Funktion, deren globales Minimum erreicht werden soll, heißt <b>Verlustfunktion / Loss-Funktion </b>. Die Funktion MSE (Mean Squared Error) ist ein Beispiel für so eine Funktion:
    
        $$MSE = \dfrac{1}{n} \bigl[ (y_1 - o_1)^2 + (y_2 - o_2)^2 + \dots + (y_n - o_n)^2 \bigr], $$
    
        wobei $(y_1, \dots, y_n)$ die optimale und $(o_1, \dots, o_n)$ die tatsächliche Ausgabe eines neuronalen Netzes beschreibt. 
    
        ____
    
        <i style="font-size:38px">?</i>
    
    
        <i>Wenn wir z.B. einen Datenpunkt betrachten, der ein Blaumeisen-Ei repräsentiert, dann ist die optimale Ausgabe bei drei möglichen Klassen (Klasse 0 = Blaumeisen, Klasse 1 = Ente, Klasse 2 = Greifvogel) der Vektor $(1, 0, 0)$. Wenn die tatsächliche Ausgabe des neuronalen Netzes $(0.5, 0.25, 0.25)$ ist, was ist dann der Verlust nach der oberen Formel?</i>
    
        <details>
    
        <summary>➤ Klicke hier, um deine Antwort zu prüfen.</summary>
    
        $$\dfrac{1}{3} \bigl[ (1 - 0.5)^2 + (0 - 0.25)^2 + (0 - 0.25)^2 \bigr] = 0.375.$$
    
        Wenn das neuronale Netz nur ein Gewicht hat, könnte die Verlustfunktion so aussehen:
    
        <figure>
          <img src="public/resources/img/loss_function2.png" alt="Verlustfunktion" style="width:45%">
        </figure> 
    
        Das aktuelle Gewicht $w_1$ von $0.7$ muss also ein bisschen vergrößert werden, um den Verlust zu verkleinern.
    
        </details>
    
        Wenn das neuronale Netz nur zwei Gewichte hat, könnte eine Verlustfunktion wie folgt aussehen. Bei mehr als zwei Gewichten (in der Praxis eingesetzte neuronale Netze haben Millionen von trainierbaren Gewichten) ist eine Visualisierung allerdings nicht mehr so einfach möglich.
    
        <figure>
          <img src="public/resources/img/train_val_loss_landscape.png" alt="Loss-Function" style="width:50%">
        </figure> 
    
        Wenn wir bestimmt haben, ob wir ein Gewicht verkleinern oder vergrößern müssen, um den Verlust zu reduzieren, müssen wir noch festlegen, wie stark wir das Gewicht verändern möchten. Dabei können unterschiedliche Probleme auftreten. Ist die Veränderung des Gewichts zu gering, kann es sein, dass das neuronale Netz in einem lokalen Minimum stecken bleibt oder sich nur sehr langsam dem globalen Minimum nähert. Verändern wir das Gewicht zu stark, ist es möglich, dass wir über das Ziel hinausschießen. 
    
        <figure>
          <img src="public/resources/img/loss_function3.png" alt="Verlustfunktion" style="width:95%">
        </figure> 
    
        Wir müssen also die <b>Lernrate</b> des neuronalen Netzes mit Bedacht wählen und möglicherweise immer wieder anpassen. Die Update-Regel für jedes Gewicht $w$ im neuronalen Netz können wir folgendermaßen notieren:
    
        $$w_{\text{neu}} \longleftarrow w_{\text{alt}} - \alpha \cdot \Delta w.$$
    
        $\alpha$ ist die Lernrate und $\Delta w$ der Gradient (die Ableitung) des Gewichts. Der Gradient gibt nicht nur die Richtung an, in der das Gewicht verändert werden muss, sondern beschreibt auch, wie stark das betrachtete Gewicht zu dem Verlust beigetragen hat. 
    
        Den Gradienten eines Gewichts $w$ bestimmen wir, indem wir die Verlustfunktion nach $w$ durch mehrfache Anwendung der Kettenregel ableiten. Da dieser Prozess sehr mühselig ist, verzichten wir an dieser Stelle auf weitere Details, weil PyTorch für uns diese Arbeit übernehmen wird.
    
        <figure>
          <img src="public/resources/img/backpropagation.png" alt="Verlustfunktion" style="width:65%">
        </figure> 
    
        Die Berechnung der Gradienten bei der Backpropagation erfordert sehr viel Rechenaufwand. Eine CPU wird nur bei kleinen Daten(mengen) gute Ergebnisse in überschaubarer Zeit liefern können. Aus diesem Grund verwendet man GPU-Einheiten (Grafikprozessoren), um ein neuronales Netz trainieren zu lassen. Der Vorteil dieser Verwendung besteht darin, dass die Berechnungen <i>parallel</i> ablaufen können und das Netz somit viel schneller trainiert.
    
        ## Training eines neuronalen Netzes
    
        Nach so viel Theorie können wir endlich neuronale Netze trainieren lassen! Untersuche den Code, um dein eigenes neuronales Netz weiter unten an die Daten anzupassen.
        """
    )
    return


@app.cell
def _():
    from public.resources.code.help_functions import daten, datenpunkte_zeichnen
    import matplotlib.pyplot as plt
    from matplotlib import colors

    (x_train, y_train, x_test, y_test) = daten()
    print(f'Wir haben {len(y_train)} Trainingsdatenpunkte und {len(y_test)} Testdatenpunkte zur Verfügung.')
    datenpunkte_zeichnen(x_train, y_train, ['#ec90cc', '#4f7087'])
    return datenpunkte_zeichnen, x_train, y_train


@app.cell
def _(mo):
    mo.md(
        r"""
        Wir implementieren folgendes neuronales Netz, das du bereits oben konstruiert hast.
    
        &nbsp;
    
    
         <figure>
          <img src="public/resources/img/nn3.png" alt="neuronales Netz" style="width:60%">
          <figcaption></figcaption>
        </figure> 
    
        &nbsp;
        """
    )
    return


@app.cell
def _(FullyConnectedLayer, relu, softmax):
    class Net_1:
        def __init__(self, num_in, num_out):
            # First fully connected layer maps from num_in to 2
            self.fc1 = FullyConnectedLayer(num_in, 2)
            # Second fully connected layer maps from 2 to 2
            self.fc2 = FullyConnectedLayer(2, 2)

        def forward(self, x):
            x = relu(self.fc1.forward(x))
            x = self.fc2.forward(x)
            x = softmax(x)
            return x

        def __str__(self):
            rep = "Net_1 architecture:\n"
            rep += "Layer 1 (fc1):\n" + str(self.fc1) + "\n"
            rep += "Layer 2 (fc2):\n" + str(self.fc2) + "\n"
            return rep
    return (Net_1,)


@app.cell
def _(Net_1):
    net = Net_1(2,2)
    print(net)
    return (net,)


@app.cell
def _(cross_entropy_loss, net, np, softmax, x_train, y_train):
    # Hyperparameters:
    epochs = 10
    lr = 0.1

    # Training loop:
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0

        # Loop over each sample; here we have just one sample.
        for x, label in zip(x_train, y_train):
            # Forward pass:
            output = net.forward(x)          # raw scores (logits)
            probs = softmax(output)            # convert scores to probabilities

            # Compute loss:
            loss = cross_entropy_loss(probs, label)
            epoch_loss += loss

            # Prediction:
            pred = np.argmax(probs)
            if pred == label:
                correct += 1

            # Compute gradient of loss with respect to logits:
            # The derivative of cross-entropy loss w.r.t. the logits (after softmax) is:
            # dL/dz = probs - y_true, where y_true is one-hot encoded.
            dout = probs.copy()
            dout[label] -= 1  # subtract 1 for the true class

            # Backward pass:
            net.fc1.backward(dout)
            net.fc2.backward(dout)
            # Update parameters:
            net.fc1.update_params(lr)
            net.fc2.update_params(lr)

        # Because we have one sample, accuracy is 100% if correct else 0%
        accuracy = (correct / len(x_train)) * 100  
        print(f"Epoch {epoch+1:3d} | Loss: {epoch_loss:.5f} | Accuracy: {accuracy:.1f}%")

    print("\nTrained network parameters:")
    print(net)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ____
    
        <img style="float: left;" src="public/resources/img/laptop_icon.png" width=50 height=50 /> <br><br>
    
        <i>Du kennst nun alle Codebausteine, um dein eigenes neuronales Netz zu konstruieren und es trainieren zu lassen. Setze ein neuronales Netz für die folgenden Daten um und passe die Gewichte an den Datensatz an. Brich das Training ab, sobald das Netz eine 93%-Genauigkeit auf dem Trainingsdatensatz erzielt. Speichere außerdem in jeder Epoche das Netz, das über alle vergangenen Durchläufe hinweg die höchste Genauigkeit erreicht hat.</i>
        """
    )
    return


@app.cell
def _(datenpunkte_zeichnen):
    from public.resources.code.help_functions import daten2
    (x_train_1, y_train_1, x_test_1, y_test_1) = daten2()
    print(f'Wir haben {len(y_train_1)} Trainingsdatenpunkte und {len(y_test_1)} Testdatenpunkte zur Verfügung.')
    datenpunkte_zeichnen(x_train_1, y_train_1, ['#ec90cc', '#8b4513', '#4f7087'])
    return


@app.cell
def _():
    # Implementiere hier die Klasse für dein neuronales Netz.
    return


@app.cell
def _():
    # Erzeuge hier das Objekt deiner Klasse und den Optimizer. 
    # Lege hier außerdem deine Loss-Funktion und die Anzahl der Epochen fest.
    return


@app.cell
def _():
    # Implementiere hier deinen Trainingsprozess.
    # Breche den Trainingsprozess ab, wenn eine Genauigkeit von 93% auf den
    # Trainingsdaten erreicht wurde.
    # Speicher außerdem immer das bisher beste Model mit deepcopy(model) 
    # in einer Variablen ab.

    from copy import deepcopy
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        <h2>Bildquellen</h2>
    
        https://pixabay.com/de/photos/ai-generiert-junge-junger-mann-7772478/
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"# Appendix")
    return


@app.cell
def _():
    import numpy as np
    np.random.seed(0)

    class FullyConnectedLayer:
        def __init__(self, input_dim, output_dim, bias=True):
            self.input_dim = input_dim
            self.output_dim = output_dim
            # Initialize weights with small random numbers.
            self.weight = np.random.randn(output_dim, input_dim) * 0.1
            self.use_bias = bias
            if self.use_bias:
                self.bias = np.random.randn(output_dim) * 0.1
            else:
                self.bias = None

            # Placeholders for gradients
            self.grad_weight = np.zeros_like(self.weight)
            if self.use_bias:
                self.grad_bias = np.zeros_like(self.bias)
            else:
                self.grad_bias = None

            # Cache for input during forward pass (needed for backprop)
            self.x = None

        def forward(self, x):
            """
            x: input array of shape (input_dim,) or (batch_size, input_dim)
            Returns:
                output: array of shape (output_dim,) or (batch_size, output_dim)
            """
            self.x = x  # cache input for use in backward pass
            y = np.dot(x, self.weight.T)  # if x is (batch,) then use (batch, output_dim)
            if self.use_bias:
                y += self.bias
            return y

        def backward(self, dout):
            """
            Computes gradients for weights, biases and returns gradient with respect to input.

            dout: Upstream gradient. Array of shape (output_dim,) or (batch_size, output_dim)

            Returns:
                dx: Gradient with respect to input x, with shape matching self.x.
            """
            # Ensure x is available
            if self.x is None:
                raise ValueError("forward must be called before backward")

            # If inputs are 1D, promote them to 2D arrays for batch processing
            single_sample = False
            if self.x.ndim == 1:
                x = self.x[None, :]   # shape: (1, input_dim)
                dout = dout[None, :]  # shape: (1, output_dim)
                single_sample = True
            else:
                x = self.x

            # Compute gradients with respect to weight and bias
            # Note: weight shape: (output_dim, input_dim)
            # x.T shape: (input_dim, batch_size) and dout shape: (batch_size, output_dim)
            # so to get grad_weight shape (output_dim, input_dim): 
            self.grad_weight = np.dot(dout.T, x) / x.shape[0]  # average over batch

            if self.use_bias:
                # Average gradient over batch
                self.grad_bias = np.mean(dout, axis=0)

            # Compute gradient with respect to input x for further backpropagation
            # x gradient: (batch, input_dim) = dout (batch, output_dim) dot weight (output_dim, input_dim)
            dx = np.dot(dout, self.weight)

            # If we processed a single sample, return as 1D vector
            if single_sample:
                dx = dx.squeeze(0)

            return dx

        def update_params(self, lr=0.1):
            """
            Update parameters using gradients computed in the backward pass.
            """
            self.weight -= lr * self.grad_weight
            if self.use_bias:
                self.bias -= lr * self.grad_bias

        def __str__(self):
            s = f'Weights shape: {self.weight.shape}\n'
            if self.use_bias:
                s += f'Bias shape: {self.bias.shape}\n'
            else:
                s += 'Bias disabled\n'
            return s

    # Example utility functions
    def relu(x):
        return np.maximum(0, x)

    def softmax(x):
        # Numerically stable softmax
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def cross_entropy_loss(probs, label):
        """
        probs: The predicted probability vector from softmax.
        label: The true class (integer)

        Returns:
            loss: Cross entropy loss (scalar)
        """
        # Add a small number to avoid log(0)
        return -np.log(probs[label] + 1e-15)
    return FullyConnectedLayer, cross_entropy_loss, np, relu, softmax


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
