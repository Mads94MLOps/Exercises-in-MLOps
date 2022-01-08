import argparse
import sys

import torch
import numpy as np

from data import mnist
from model import MyAwesomeModel
from torch import optim
from torch import nn



class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        trainloader, _ = mnist()
        
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003)


        epochs = 5
        
        print('Before training loop')
        for i in range(epochs):
            running_loss = 0
            for images, labels in trainloader:
                
                images = images.view(images.shape[0], -1)
                
                optimizer.zero_grad()
                #import pdb
                #pdb.set_trace()
                output = model(images)
                loss = criterion(output,labels)
                loss.backward()
                
                optimizer.step()
                
                running_loss += loss.item()
                
            else:
                print(f'Traning loss: {running_loss/len(trainloader)}')
        
        #torch.save(model, "C:/Users/Mads_/OneDrive/Anvendt Kemi/Machine Learning Operations/dtu_mlops/s1_getting_started/exercise_files/final_exercise/s1-M4_Pytorch/training_model.pt")
        torch.save(model.state_dict(), "C:/Users/Mads_/OneDrive/Anvendt Kemi/Machine Learning Operations/dtu_mlops/s1_getting_started/exercise_files/final_exercise/s1-M4_Pytorch/training_model.pt")
        
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        #model = torch.load(args.load_model_from)
        model = MyAwesomeModel()
        model.load_state_dict(torch.load(args.load_model_from))
        model.eval()
        _, testloader = mnist()
        
        with torch.no_grad():
            #model.eval()
            for images, labels in testloader:
                images = images.view(images.shape[0], -1)
                log_ps = model(images)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy = torch.mean(equals.type(torch.FloatTensor))

        print(f'Accuracy: {accuracy.item()*100}%')
        
        

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    