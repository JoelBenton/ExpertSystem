import os
import numpy as np

File = open('phpgNaXZe.txt','r')
TestDate = File.readlines()

# Sigmoid function for output.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


Train_X_Array = []
Train_ExpectedResults = []

Test_X_Array = []
Test_ExpectedResults = []

# Format Data - Start

loopNum = 1
## Generate X values and Expected Results - Start
for data in TestDate:
    data = data.split(",")
    X1 = float(data[0])
    X2 = float(data[1])
    X3 = float(data[2])
    X4 = float(data[3])
    X5 = float(data[4])
    X6 = float(data[5])
    X7 = float(data[6])
    X8 = float(data[7])
    X9 = float(data[8])
    X10 = int(data[9])
    
    if loopNum <= 400:
        Train_ExpectedResults.append(X10)
        X = [float(X1), float(X2), float(X3), float(X4), float(X5), float(X6),float(X7), float(X8), float(X9)]
        Train_X_Array.append(X)
    else:
        Test_ExpectedResults.append(X10)
        X = [float(X1), float(X2), float(X3), float(X4), float(X5), float(X6),float(X7), float(X8), float(X9)]
        Test_X_Array.append(X)
        
    loopNum = loopNum + 1

## Formatting Expected Results - Start

Train_ExpectedResults = [0 if x == 1 else 1 if x == 2 else x for x in Train_ExpectedResults]
Test_ExpectedResults = [0 if x == 1 else 1 if x == 2 else x for x in Test_ExpectedResults]

## - End
# - End

# INTIALISING VARIABLES - Start

Total_Epoch = 10

Input_Nodes = 9
Hidden_Nodes = 5
Output_Nodes = 1

Total_Weights = 50

Batch_Size = 10
Learning_Curve = 0.1

epsilon = 1e-15

Hidden_Bias = [0 for i in range(Hidden_Nodes)]

Output_Bias = [0 for i in range(Output_Nodes)]

Hidden_Nodes_Output = []
Output_Nodes_Output = []


# End

# INTIALISING WEIGHTS USING Xavier Initalization - Start

Hidden_Node_1_Weights = [np.random.randn(Input_Nodes, Output_Nodes) * np.sqrt(2.0/(Input_Nodes + Output_Nodes))]
Hidden_Node_2_Weights = [np.random.randn(Input_Nodes, Output_Nodes) * np.sqrt(2.0/(Input_Nodes + Output_Nodes))]
Hidden_Node_3_Weights = [np.random.randn(Input_Nodes, Output_Nodes) * np.sqrt(2.0/(Input_Nodes + Output_Nodes))]
Hidden_Node_4_Weights = [np.random.randn(Input_Nodes, Output_Nodes) * np.sqrt(2.0/(Input_Nodes + Output_Nodes))]
Hidden_Node_5_Weights = [np.random.randn(Input_Nodes, Output_Nodes) * np.sqrt(2.0/(Input_Nodes + Output_Nodes))]

Hidden_Node_Weights = [Hidden_Node_1_Weights, Hidden_Node_2_Weights, Hidden_Node_3_Weights, Hidden_Node_4_Weights, Hidden_Node_5_Weights]

Output_Node_1_Weights = [(np.random.randn(Input_Nodes, Output_Nodes) * np.sqrt(2.0/(Input_Nodes + Output_Nodes)))]

Output_Node_Weights = [Output_Node_1_Weights]

# - End

print()

while (True):
    # Will loop through the test data for the number of Epochs specified.
    for i in range(Total_Epoch):
        print("Epoch " , i + 1)
        
        Expected_Result_Batch = []
        
        Total_Epoch_Lost = []
        
        # Will loop through each test data no matter the amount
        for a in range(len(Train_X_Array)):

            Expected_Result_Batch.append(Train_ExpectedResults[a])
            
            Training_Data = Train_X_Array[a]
            
            for b in range(Hidden_Nodes):
                Weights = Hidden_Node_Weights[b]
                
                y = np.maximum(0, ((sum((x * y for x, y in zip(Weights[0], Training_Data))) + Hidden_Bias[b])))
                
                Hidden_Nodes_Output.append(y)
                
                
            for c in range(Output_Nodes):
                Weights = Output_Node_Weights[c]
                Data = Hidden_Nodes_Output
                
                y = sigmoid((sum((x * y for x, y in zip(Weights[0], Data))) + Output_Bias[c]))
                
                Output_Nodes_Output.append(y)

            # After every 10 Training data, it will run the the Loss calculations and update the weights and bias.
            if ((a + 1) % Batch_Size == 0 and a != 0):
                
                # Training Starting at every 10th cycle.
                
                ## Output Weights Training.
                
                for d in range(Output_Nodes): 
                    
                    # Calculate loss for batch
                    loss = []
                    
                    for e in range(10):
                        Output_Nodes_Output[e] = np.clip(Output_Nodes_Output[e], epsilon, 1 - epsilon)
                        loss = -(Expected_Result_Batch[e] * np.log(Output_Nodes_Output) + (1 - Expected_Result_Batch[e]) * np.log(1 - Output_Nodes_Output[e]))
                        
                    #Total_Loss = np.sum(loss)
                    
                    Average_Loss = np.atleast_1d(np.mean(loss))
                    
                Hidden_Nodes_Output = []
                Output_Nodes_Output = []
                Expected_Result_Batch = []

        
        
    # Run Tests Again!
    Run_Again = input("Loop Again? N to exit (case sensitive): ")
    if (Run_Again == "N"):
        break