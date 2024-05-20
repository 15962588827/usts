Predictive control optimization of household energy storage devices for load regulation and energy conservation 


The paper involves the following code, including the specific model and household environment of the prediction method SA-LSTM model and the control method DQN.

1.LSTM_main is the learning process of SA-LSTM,LSTM_models  is the network definition of SA-LSTM and LSTM utils is the sliding Windows .

2.HEMS1 is the interaction process between DQN and household environment

3.env1 models the home energy management system

In this study, the prediction of the load of the house load is predicted, and the result of the prediction is passed to the agent as an element of the state, and the optimal control policy of the energy storage device is obtained by the DQN , and the goal of the household load control and energy saving is realized.

