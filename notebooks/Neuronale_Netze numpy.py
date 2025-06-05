import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.accordion(
                {
                    "🗬 Neuronale Netze": r"""
    # Neuronale Netze

    Ein Neuronales Netz besteht aus Schichten aus Neuronen. Ein einzelnes Neuron, auch als Perzeptron bezeichnet, besteht aus ...

    - einer festen Anzahl <b>Inputs</b> (abhängig von den Dimensionen der Punkte, die als Datengrundlage dienen)
    - <b>Gewichten</b> mit denen die Eingaben mulipliziert 
    - und zusammen mit dem <b>Bias</b> addiert werden und 
    - einer <b>Aktivierungsfunktion</b>. 

    Diesen Aufbau bezeichnen wir im Folgenden als <b>Neuron</b>.

    <figure>
      <img src="public/resources/img/perzeptron.png" alt="perzeptron" style="width:70%">
    </figure> 

    ## Aufbau neuronaler Netze

    Im Folgenden ändern wir das Perzeptron Schritt für Schritt ab, um dessen Defizite zu beheben.

    ### Mehr als zwei Klassen klassifizieren und Performance steigern

    Um die Performance unserer KI zu steigern, schalten wir mehrere Neuronen hinter- und nebeneinander. Die Ausgabe eines Neurons dient nun als Eingabe von nachfolgenden Neuronen. Sind Neuronen parallel in einer Ebene angeordnet, wird die Gesamtheit dieser Neuronen als <b>Layer</b> (bzw. Schicht) bezeichnet. Das gesamte Konstrukt mehreren Neuronenschichten bezeichnet man als <b>neuronales Netz</b>. Wenn es mehrere verdeckte Schichten gibt, bezeichnet man das Netz als <b>tiefes neuronales Netz</b> (deep neural network).

    <figure>
      <img src="public/resources/img/nn1.png" alt="perzeptron" style="width:60%">
    </figure> 

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

    <figure>
      <img src="public/resources/img/sigmoid_and_relu.png" alt="Sigmoid and ReLU" style="width:50%">
    </figure> 

    ### Softmax

    Jetzt fehlt nur noch eine kleine Änderung, um ein herkömmliches neuronales Netz zu erhalten. Wie im vorletzten Abschnitt bereits umrissen, wird die Klassifikation des Datenpunkts jetzt nicht mehr durch eine 0- oder 1-Ausgabe des letzten Neurons ermittelt, sondern durch die Nummer des Neurons in der Ausgabeschicht, das die größte Ausgabe hat. Durch die neue ReLU-Aktivierungsfunktion erhalten wir in der letzten Ausgabeschicht nicht mehr 0- oder 1-Ausgaben, sondern Werte größer oder gleich 0. 
    Um als Ausgabe des neuronalen Netzes die Wahrscheinlichkeit zu erhalten, mit der ein Datenpunkt einer Klasse zugeordnet wird, wird eine am Ende eine zusätzliche Schicht mit einer speziellen Aktivierungsfunktion (Softmax-Funktion) eingefügt, deren Gewichte nicht trainiert werden.
    """
                }
            ),
            mo.md(
                r"""
    <figure>
      <img src="public/resources/img/nn2.png" alt="perzeptron" style="width:80%">
    </figure> 

    Jetzt sind wir bereit unser erstes neuronales Netz in Code umzusetzen. Damit wir nicht alles selbst implementieren müssen, verwenden wir die Funktionen `FullyConnectedLayer()` sowie die `relu()` und `softmax()` Aktivierungsfunktion. Diese verwendenen die Software Library <i>Numpy</i>.

    ## Numpy

    Numpy ermöglicht es auf eine sehr einfache Weise, neuronale Netze zu konstruieren. Gehe das folgende Codefeld durch und führe es aus, um mit den Funktionsaufrufen vertraut zu werden. Wir konstruieren dabei das obige neuronale Netze mit vier Eingabe- und drei Ausgabeneuronen.
    """
            ),
        ]
    )
    return


@app.cell
def _(FullyConnectedLayer, np, relu, softmax):
    class Net:
        # Im Konstruktor werden die unterschiedlichen Schichten definiert
        def __init__(self, num_in, num_out):
            self.name_model = "Netzi"

            # Durch den folgenden Funktionsaufruf wird eine Schicht mit num_in eingehenden 
            # und 5 ausgehenden Verbindungen konstruiert.
            # Den Namen der Schichten kannst du selbst festlegen.
            # fc steht für fully connected.
            self.fc1 = FullyConnectedLayer(num_in, 5)  
            # Achte darauf, dass die folgende Schicht die Anzahl der eingehenden Verbindungen
            # mit den ausgehenden Verbindungen der letzten Schicht übereinstimmt.
            self.fc2 = FullyConnectedLayer(5, 5)  # second layer: 5 ->5
            # Standardmäßig wird zu jedem Neuron ein Bias hinzugefügt. Durch den Parameter
            # 'bias' kann das deaktiviert werden.
            self.fc3 = FullyConnectedLayer(
                5, num_out, bias=False
            )

        # In dieser Funktion muss festgelegt werden, wie die Eingabe durch das Netz propagiert wird (d.h. durch
        # die einzelnen Schichten „weitergereicht“ wird).
        def forward(self, x):
            # Zunächst wird die Eingabe mit den Gewichten der ersten Schicht multipliziert, 
            # in den einzelnen Neuronen aufsummiert und anschließend in die ReLU-Funktion eingesetzt.
            output = relu(self.fc1.forward(x))
            # Die verarbeitete Eingabe wird nun durch die zweite Schicht propagiert. 
            output = relu(self.fc2.forward(output))
            # In der vorletzten Schicht gibt es keine ReLU-Funktion mehr.
            output = self.fc3.forward(output)
            # finale Ausgabe des neuronalen Netzes
            output = softmax(output)
            return output

        def __str__(self):
            # Um beim Aufruf print() die Parameter ablesen zu können
            rep = f"Model name: {self.name_model}\n"
            rep += "Layer 1 (fc1):\n" + str(self.fc1) + "\n"
            rep += "Layer 2 (fc2):\n" + str(self.fc2) + "\n"
            rep += "Layer 3 (fc3):\n" + str(self.fc3)
            return rep


    # Erzeugung eines Objekts des neuronalen Netzes
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

    # Das ist eine Testeingabe
    test_eingabe = np.array([1.0, 2.5, -1, 0])

    # Die Ausgabe erhälst du so
    ausgabe = erstes_nn.forward(test_eingabe)
    print("\nAusgabe (Ergebnis des Vorwärtsdurchlaufs):", ausgabe)

    return (erstes_nn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ____
    # Aufgabe 1
    <img style="float: left;" src="public/resources/img/laptop_icon.png" width=50 height=50 /> <br><br>

    <i>Im letzten Codefeld wurden unser erstes neuronales Netz erzeugt. Lies die gesuchten Gewichte anhand der letzten Ausgabe ab und überprüfe deine Eingabe, indem du das Codefeld ausführst. Runde gegebenenfalls die Eingaben auf die vierte Nachkommastelle ab.</i>
    """
    )
    return


@app.cell
def _(erstes_nn, pruefe_gewichte):
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ____
    # Aufgabe 2
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


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Tipp 1": "Sieh dir nochmal die Definition von `class Net` an.",
            "Tipp 2": "Du brauchst um ein eigenes neuronales Netz zu implementieren eine eigene Klasse mit 2 Eingaben und 2 Ausgaben sowie 2 versteckten Schichten",
            "Lösung": mo.md(
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
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
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
    """
            ),
            mo.accordion(
                {
                    r"Klicke hier, um deine Antwort zu prüfen.": r"""$$\dfrac{1}{3} \bigl[ (1 - 0.5)^2 + (0 - 0.25)^2 + (0 - 0.25)^2 \bigr] = 0.375.$$

    Wenn das neuronale Netz nur ein Gewicht hat, könnte die Verlustfunktion so aussehen:

    <figure>
      <img src="public/resources/img/loss_function2.png" alt="Verlustfunktion" style="width:45%">
    </figure> 

    Das aktuelle Gewicht $w_1$ von $0.7$ muss also ein bisschen vergrößert werden, um den Verlust zu verkleinern."""
                }
            ),
            mo.md(r"""
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
    """),
        ]
    )
    return


@app.cell
def _(daten, datenpunkte_zeichnen):
    (x_train, y_train, x_test, y_test) = daten()
    print(
        f"Wir haben {len(y_train)} Trainingsdatenpunkte und {len(y_test)} Testdatenpunkte zur Verfügung."
    )
    datenpunkte_zeichnen(x_train, y_train, ["#ec90cc", "#4f7087"])
    return x_test, x_train, y_test, y_train


@app.cell(hide_code=True)
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
    class Net_2:
        def __init__(self, num_in, num_out):
            # First fully connected layer maps from num_in to 2
            self.fc1 = FullyConnectedLayer(num_in, 2)
            # Second fully connected layer maps from 2 to 2
            self.fc2 = FullyConnectedLayer(2, num_out)

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
    return (Net_2,)


@app.cell
def _(Net_2):
    net = Net_2(2, 2)
    print(net)
    return (net,)


@app.cell
def _(
    cross_entropy_loss,
    evaluate_model,
    net,
    np,
    softmax,
    x_test,
    x_train,
    y_test,
    y_train,
):
    # Hyperparameter:
    epochs = 10  # Anzahl der Epochen (Durchläufe)
    lr = 0.1     # Lernrate

    # Berechnung der Genauigkeit auf den Testdaten vor dem Training
    print(
        f"Testgenauigkeit vor dem Training: {evaluate_model(net, x_test, y_test):.1f}%"
    )

    # Trainingsschleife:
    for epoch in range(epochs):  # Für jede Epoche
        epoch_loss = 0.0  # Verlust für die aktuelle Epoche
        correct = 0       # Anzahl der richtigen Vorhersagen

        # Schleife über jedes Beispiel; hier haben wir nur ein Beispiel.
        for x, label in zip(x_train, y_train):  # x_train sind die Eingabedaten, y_train die Labels
            # Vorwärtsdurchlauf:
            output = net.forward(x)  # rohe Werte (Logits)
            probs = softmax(output)  # Werte in Wahrscheinlichkeiten umwandeln

            # Verlust berechnen:
            loss = cross_entropy_loss(probs, label)  # Verlustfunktion
            epoch_loss += loss  # Verlust zur Gesamtsumme hinzufügen

            # Vorhersage:
            pred = np.argmax(probs)  # Vorhersage basierend auf den Wahrscheinlichkeiten
            if pred == label:  # Wenn die Vorhersage korrekt ist
                correct += 1  # Zähle die richtige Vorhersage

            # Gradient des Verlusts bezüglich der Logits berechnen:
            # Die Ableitung des Kreuzentropieverlusts bezüglich der Logits (nach Softmax) ist:
            # dL/dz = probs - y_true, wobei y_true als One-Hot kodiert ist.
            dout = probs.copy()  # Kopie der Wahrscheinlichkeiten
            dout[label] -= 1  # 1 für die wahre Klasse abziehen

            # Rückwärtsdurchlauf:
            net.fc1.backward(dout)  # Rückpropagation für die erste Schicht
            net.fc2.backward(dout)  # Rückpropagation für die zweite Schicht
            # Parameter aktualisieren:
            net.fc1.update_params(lr)  # Parameter der ersten Schicht aktualisieren
            net.fc2.update_params(lr)  # Parameter der zweiten Schicht aktualisieren

        # Da wir nur ein Beispiel haben, ist die Genauigkeit 100%, wenn korrekt, sonst 0%
        accuracy = (correct / len(x_train)) * 100  # Genauigkeit berechnen
        # Berechnung der Genauigkeit auf den Testdaten
        test_accuracy = evaluate_model(net, x_test, y_test)
        print(
            f"Epoche {epoch + 1:3d} | Verlust: {epoch_loss:.5f} | Genauigkeit: {accuracy:.1f}% | Testgenauigkeit: {test_accuracy:.1f}%"
        )

    # Berechnung der Genauigkeit auf den Testdaten nach dem Training
    print(
        f"Testgenauigkeit nach dem Training: {evaluate_model(net, x_test, y_test):.1f}%"
    )

    print("\nTrainierte Netzwerkparameter:")
    print(net)  # Ausgabe der trainierten Netzwerkparameter

    return


@app.cell
def _(np):
    # Funktion zur Berechnung der Genauigkeit auf den Testdaten
    def evaluate_model(net, x_test, y_test):
        correct = 0  # Zähler für richtige Vorhersagen

        # Schleife über jedes Testbeispiel
        for x, label in zip(x_test, y_test):
            # Vorwärtsdurchlauf
            output = net.forward(x)  # Wahrscheinlichkeiten für jede Klasse
            pred = np.argmax(output)  # Vorhersage basierend auf den Wahrscheinlichkeiten

            # Überprüfen, ob die Vorhersage korrekt ist
            if pred == label:
                correct += 1  # Zähle die richtige Vorhersage

        # Berechnung der Genauigkeit
        accuracy = (correct / len(x_test)) * 100  # Genauigkeit in Prozent
        return accuracy

    return (evaluate_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ____
    # Aufgabe 3
    <img style="float: left;" src="public/resources/img/laptop_icon.png" width=50 height=50 /> <br><br>

    <i>Du kennst nun alle Codebausteine, um dein eigenes neuronales Netz zu konstruieren und es trainieren zu lassen. Setze ein neuronales Netz für die folgenden Daten um und passe die Gewichte an den Datensatz an. Brich das Training ab, sobald das Netz eine 93%-Genauigkeit auf dem Trainingsdatensatz erzielt. Speichere außerdem in jeder Epoche das Netz, das über alle vergangenen Durchläufe hinweg die höchste Genauigkeit erreicht hat.</i>
    """
    )
    return


@app.cell
def _(daten2, datenpunkte_zeichnen):
    (x_train_1, y_train_1, x_test_1, y_test_1) = daten2()
    print(
        f"Wir haben {len(y_train_1)} Trainingsdatenpunkte und {len(y_test_1)} Testdatenpunkte zur Verfügung."
    )
    datenpunkte_zeichnen(x_train_1, y_train_1, ["#ec90cc", "#8b4513", "#4f7087"])
    return


@app.cell
def _():
    # Implementiere hier die Klasse für dein neuronales Netz.
    return


@app.cell
def _():
    # Erzeuge hier das Objekt deiner Klasse.
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
    mo.md(r"""# Appendix""")
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
            y = np.dot(
                x, self.weight.T
            )  # if x is (batch,) then use (batch, output_dim)
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
                x = self.x[None, :]  # shape: (1, input_dim)
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
            s = f"Weights shape: {self.weight.shape}\n"
            if self.use_bias:
                s += f"Bias shape: {self.bias.shape}\n"
            else:
                s += "Bias disabled\n"
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
def _(np):
    import matplotlib.pyplot as plt
    from matplotlib import colors


    def pruefe_gewichte(nn, gewicht1, gewicht2, gewicht3, gewicht4):
        """
        Parameters:
        nn: A network object with attributes fc1, fc2, fc3, whose parameters are NumPy arrays.
        gewicht1: Expected value for fc1.weight[0][0]
        gewicht2: Expected value for fc1.bias[4]
        gewicht3: Expected value for fc2.bias[1]
        gewicht4: Expected value for fc3.weight[2][3]

        Returns:
          A string summarizing whether each weight has been read correctly.
        """
        # For instance, nn.fc1.weight is assumed to be a NumPy array.
        g1 = round(nn.fc1.weight[0][0], 4)
        g2 = round(nn.fc1.bias[4], 4)
        g3 = round(nn.fc2.bias[1], 4)
        g4 = round(nn.fc3.weight[2][3], 4)

        wrong = False
        result = ""

        if g1 == round(gewicht1, 4):
            result = "Das erste Gewicht hast du richtig abgelesen!\n"
        else:
            result = "Das erste Gewicht hast du nicht richtig abgelesen!\n"
            wrong = True

        if g2 == round(gewicht2, 4):
            result += "Das zweite Gewicht hast du richtig abgelesen!\n"
        else:
            result += "Das zweite Gewicht hast du nicht richtig abgelesen!\n"
            wrong = True

        if g3 == round(gewicht3, 4):
            result += "Das dritte Gewicht hast du richtig abgelesen!\n"
        else:
            result += "Das dritte Gewicht hast du nicht richtig abgelesen!\n"
            wrong = True

        if g4 == round(gewicht4, 4):
            result += "Das vierte Gewicht hast du richtig abgelesen!\n"
        else:
            result += "Das vierte Gewicht hast du nicht richtig abgelesen!\n"
            wrong = True

        if wrong:
            return result
        else:
            return "Super! Du hast alle Gewichte richtig abgelesen!"


    def daten():
        """
        Returns:
        x_train: A NumPy array of training input data of shape (400, 2)
        y_train: A NumPy array of training labels of shape (400,)
        x_test:  A NumPy array of test input data of shape (100, 2)
        y_test:  A NumPy array of test labels of shape (100,)
        """
        # Create base data (all ones)
        n_data_train = np.ones((200, 2))
        n_data_test = np.ones((50, 2))

        # Create training data:
        # Class 0: Normal distribution centered at [2.5, 5] (for all 200 samples)
        x0 = np.random.normal(loc=n_data_train + np.array([2.5, 5]), scale=1.0)
        y0 = np.zeros(200, dtype=int)
        # Class 1: Normal distribution centered at [8, 2]
        x1 = np.random.normal(loc=n_data_train + np.array([8, 2]), scale=1.0)
        y1 = np.ones(200, dtype=int)

        x_train = np.concatenate((x0, x1), axis=0).astype(np.float32)
        y_train = np.concatenate((y0, y1), axis=0)

        # Create test data:
        n_data_test = np.ones((50, 2))
        x0_test = np.random.normal(loc=n_data_test + np.array([2.5, 5]), scale=1.0)
        y0_test = np.zeros(50, dtype=int)
        x1_test = np.random.normal(loc=n_data_test + np.array([8, 2]), scale=1.0)
        y1_test = np.ones(50, dtype=int)

        x_test = np.concatenate((x0_test, x1_test), axis=0).astype(np.float32)
        y_test = np.concatenate((y0_test, y1_test), axis=0)

        return x_train, y_train, x_test, y_test


    def datenpunkte_zeichnen(x_data, labels, farben):
        """
        Parameters:
          x_data: A NumPy array with the data points, shape (N, 2)
          labels: A NumPy array with the class labels of each data point.
          farben: A list of color names for the classes.
        """
        plt.scatter(
            x_data[:, 0],
            x_data[:, 1],
            c=labels,
            s=50,
            cmap=colors.ListedColormap(farben),
        )
        plt.show()


    def daten2():
        """
        Returns:
        x_train: A NumPy array of training input data of shape (600, 2)
        y_train: A NumPy array of training labels of shape (600,)
        x_test:  A NumPy array of test input data of shape (150, 2)
        y_test:  A NumPy array of test labels of shape (150,)`
        """
        # Training data:
        n_data_train = np.ones((200, 2))

        x0 = np.random.normal(loc=n_data_train + np.array([6, 5]), scale=1.0)
        y0 = np.zeros(200, dtype=int)
        x1 = np.random.normal(loc=n_data_train + np.array([2, 2]), scale=1.0)
        y1 = np.ones(200, dtype=int)
        x2 = np.random.normal(loc=n_data_train + np.array([10, 2]), scale=1.0)
        y2 = 2 * np.ones(200, dtype=int)

        x_train = np.concatenate((x0, x1, x2), axis=0).astype(np.float32)
        y_train = np.concatenate((y0, y1, y2), axis=0)

        # Test data:
        n_data_test = np.ones((50, 2))

        x0_test = np.random.normal(loc=n_data_test + np.array([6, 5]), scale=1.0)
        y0_test = np.zeros(50, dtype=int)
        x1_test = np.random.normal(loc=n_data_test + np.array([2, 2]), scale=1.0)
        y1_test = np.ones(50, dtype=int)
        x2_test = np.random.normal(loc=n_data_test + np.array([10, 2]), scale=1.0)
        y2_test = 2 * np.ones(50, dtype=int)

        x_test = np.concatenate((x0_test, x1_test, x2_test), axis=0).astype(
            np.float32
        )
        y_test = np.concatenate((y0_test, y1_test, y2_test), axis=0)

        return x_train, y_train, x_test, y_test
    return daten, daten2, datenpunkte_zeichnen, pruefe_gewichte


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
