# Repo per il progetto finale di Applied Machine Learning AA 24-25

Il progetto finale di Applied Machine Learning consiste nell'applicare le conoscenze ottenute durante le lezioni, quindi scegliere un database e farne analisi, visualizzazione dati e poi provare diverse tipologie di modello in modo da determinare il migliore in termini di una serie di metriche.

Il nostro gruppo è composto da Tiziano Costa e Leonardo Parise e intendiamo studiare dati relativi lo stato di salute di turbine di un elicottero. 
Per farlo abbiamo scelto di utilizzare modelli di classificazione per determinare la presenza di guasto o meno.
Proveremo ad applicare la regressione logistica e decision tree.

## Motivazione

Nell’industria il guadagno è determinato dalla capacità produttiva oraria, riuscire quindi a garantire un’alta percentuale di funzionamento del macchinario nel periodo è cruciale.
Capita però che si verifichino dei guasti che costringono a interrompere le operazioni e richiedere intervento di tecnici.
Non sempre però è facile determinare quando un macchinario è guasto, e l’utilizzo in questa condizione potrebbe aggravarne ulteriormente lo stato. La capacità di rilevamento dei guasti diventa perciò cruciale.
Nei sistemi moderni, la disponibilità di sensori permette di conoscere lo stato di operazione di un macchinario, salvare questi dati permette di ottenere diversi punti di lavoro, che possono quindi essere analizzati da modelli di machine learning per rilevare guasti in modo affidabile e rapido.
Nel caso dell’applicazione in esame la determinazione del guasto in fase di volo potrebbe permettere al pilota di effettuare un atterraggio di emergenza salvando le vite dei passeggeri e il velivolo.

## Step di analisi

1. Comprensione del problema
2. Recupero dati
3. Esplorazione dei dati per maggiore comprensione del problema
4. Preparazione dei dati per evidenziare i vari aspetti significativi utili per l'analisi con algoritmi di Machine Learning
5. Analisi di vari modelli di machine learning per trovare il migliore
6. Affinare il/i modelli migliori per ottenere la miglior soluzione
7. Presentazione della soluzione

I primi 4 punti sono i più critici, riuscire ad impostare correttamente il problema è cruciale per ottenere un risultato valido.

## Metriche utilizzate

1.​ Precisione: Indica quanto i risultati sono interessanti e rilevanti, vorremmo privilegiare questo indice, valore atteso >90%
2.​ Accuratezza: Percentuale di valori correttamente classificati sul totale, valore atteso 85%
3.​ Sensitività: Percentuale di positivi correttamente classificati, questo indice è in trade off con la precisione, valore atteso 80%
4.​ Specificità: Percentuale di negativi correttamente classificati, valore atteso 85%
5.​ Confusion matrix, per visualizzazione di bontà di predizione, attesa matrice quasi diagonale
6.​ Curva Reciever Operating Characteristics e Area Under the Curve, valore atteso di AUC 80%

### Regressione Logistica

1.​ Log-loss: funzione da minimizzare per raggiungere buone performance del modello.
2.​ Gradient descent e stochastic descent: algoritmi implementabili per minimizzare la log-loss function.
3.​ odds-ratio: utile per analizzare la correlazione tra l’output e i predittori.
4.​ p value: usato per verificare che il predittore abbia un impatto sull’output. Se l’indice è inferiore a 0.05 significa che non c’è correlazione.

### Decision Tree

1.​ Gini Index come criterio di splitting
2.​ Cost Complexity Pruning per evitare la possibilità di overfitting
3.​ Parametri statici di stop rule per decision tree, quali:
  a.​ Max Depth
  b.​ Min Sample Split
  c.​ Min Sample Leaf
  d.​ Max Leaf Node
4.​ Bagging per migliorare la varianza, di cui sarà verificata la bontà utilizzando Out Of Bag error
5.​ In base ai risultati ottenuti con bagging potrei valutare di utilizzare Random Forest e verificare la bontà con OOB error
