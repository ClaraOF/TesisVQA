# Building Model's Architecture
#Now, let's build our model's architecture according to the paper. We will use PyTorch to build our model as we said before.

# For Building the model, we will use PyTorch and its functions
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sentence_transformers import SentenceTransformer, util
import numpy as np
from tqdm.notebook import tqdm
# For evaluation, we will need sklearn.metrics.average_precision_score
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from PIL import Image
# For taking the image from the URL, we will use requests
import requests

class VQAModel(nn.Module):
    #,IMAGE_MODEL = 'clip-ViT-B-32'
    def __init__(self, num_classes, hidden_size, img_model ='clip-ViT-B-32',TEXT_MODEL = 'sentence-transformers/clip-ViT-B-32-multilingual-v1', device = torch.device("cpu")):
        super(VQAModel, self).__init__()

        self.training_losses = []
        self.validation_losses = []

        self.training_accuracies = []
        self.validation_accuracies = []

        self.vizwiz_training_accuracies = []
        self.vizwiz_validation_accuracies = []

        self.training_answerability = []
        self.validation_answerability = []

        self.device = device
        self.image_name = img_model
        self.text_name = TEXT_MODEL
        

        # Initializing Binary Cross Entropy Loss which will be used to train the model on answerability
        self.answerability_loss_fn = nn.BCELoss()

        # Loading the CLIP model
        #self.clip_img_model, self.preprocess = clip.load(model_name, device = device)

        # Freezing the CLIP model
        #for param in self.clip_img_model.parameters():
        #    param.requires_grad = False

        # We use the original clip-ViT-B-32 for encoding images
        self.clip_img_model = SentenceTransformer(img_model, device)        
        print('image model: ',img_model)
        print(self.clip_img_model, self.clip_img_model.parameters() )

        # Our text embedding model is aligned to the img_model and maps 50+
        # languages to the same vector space
        self.clip_text_model = SentenceTransformer(TEXT_MODEL, device)
        print('text model')
        print(self.clip_text_model)
        print(self.clip_text_model.max_seq_length)

        # Freezing the CLIP models
        for param in self.clip_img_model.parameters():
            param.requires_grad = False

        for param in self.clip_text_model.parameters():
            param.requires_grad = False

        #print('self.clip_img_model.output_dim: ',self.clip_img_model.output_dim )
        #print('self.clip_text_model.text_projection: ',self.clip_text_model.text_projection.shape[1])
        print('hidden_size: ', hidden_size)
        print('num_classes: ',num_classes) 

        self.text_output_dim =512 # 768 xq el output de este clip es de 512
        self.visual_output_dim = 512 #768
        print('self.visual_output_dim: ', self.visual_output_dim)

        ###HAY que cambiar el hidden_size ya que ahora tengo los inputs mas chicos???????

        # First linear layer
        self.linear_layer1 = nn.Sequential(
            #nn.LayerNorm(self.clip_img_model.output_dim + self.clip_text_model.text_projection.shape[1]),
            nn.LayerNorm(self.visual_output_dim + self.text_output_dim),
            nn.Dropout(p=0.5),
            nn.Linear(self.visual_output_dim + self.text_output_dim, hidden_size)
        )

        # Second linear layer
        self.linear_layer2 = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, num_classes)
        )
        
        #ahora tengo las 4 categorias
        self.answer_type_layer = nn.Linear(hidden_size, 4)
        self.answer_mask_layer = nn.Linear(4, num_classes)

        #lo reemplace xq mi archivo solo tiene 3 tipos de rtas.
        #self.answer_type_layer = nn.Linear(hidden_size, 3)
        #self.answer_mask_layer = nn.Linear(3, num_classes)

        self.sigmoid = nn.Sigmoid()

        # Answerability Linear Layer (We removed drop out layer because training answerability was very bad)
        self.answerability_linear_layer = nn.Sequential(
            nn.LayerNorm(self.visual_output_dim + self.text_output_dim),
            nn.Linear(self.visual_output_dim + self.text_output_dim, hidden_size)
        )

        # Answerability Sigmoid Layer
        self.answerability_final_layer = nn.Linear(hidden_size, 1)

        # Sigmoid Layer for Answerability
        self.answerability_sigmoid = nn.Sigmoid()

    def forward(self, image, question):
        #print('FORWARD')
        #print('image size: ', image.size())
        #print('question size: ',question.size())

        # Flattening and concatenating the image and question features
        image = torch.flatten(image, start_dim=1)
        question = torch.flatten(question, start_dim=1)
        #print('image size after flatten: ', image.size())
        #print('question size after flatten: ',question.size())
        features = torch.cat((image, question), dim=1)
        #print('features size: ', features.size())

        # Calculating the answerability score
        answerability_score = self.answerability_linear_layer(features)
        answerability_score = self.answerability_final_layer(answerability_score)
        answerability_score = self.answerability_sigmoid(answerability_score)
        answerability_score = answerability_score.squeeze()

        # Passing the features through the first linear layer
        features = self.linear_layer1(features)
        #print('features: ', features)

        # Passing the features to get 4 answer types
        answer_type = self.answer_type_layer(features)

        # Expanding answer make to the same size as the number of classes (vocab size)
        answer_mask = self.answer_mask_layer(answer_type)

        # Applying sigmoid to get the answer mask
        answer_mask = self.sigmoid(answer_mask)

        # Passing the features through the second linear layer
        output = self.linear_layer2(features)

        # Applying the answer mask to the output
        output = output * answer_mask

        return output, answer_type, answerability_score

    def train_model(self, training_dataloader, validation_dataloader, test_dataloader, criterion, optimizer, epochs = 10, save_path = None, save_every = 1):
        for epoch in range(1,epochs+1):
            training_loss, training_accuracy, training_vizwiz_accuracy, train_answerability_score = self.training_step(training_dataloader, criterion, optimizer, self.device)
            validation_loss, validation_accuracy, validation_vizwiz_accuracy, validation_answerability_score = self.validation_step(validation_dataloader, criterion, self.device)
            test_accuracy, test_vizwiz_accuracy, test_answerability_score = self.test_step(test_dataloader)

            self.training_losses.append(training_loss)
            self.validation_losses.append(validation_loss)

            self.training_accuracies.append(training_accuracy)
            self.validation_accuracies.append(validation_accuracy)

            self.vizwiz_training_accuracies.append(training_vizwiz_accuracy)
            self.vizwiz_validation_accuracies.append(validation_vizwiz_accuracy)

            self.training_answerability.append(train_answerability_score)
            self.validation_answerability.append(validation_answerability_score)


            print("Epoch: {} | Training Loss: {:.3f} | Validation Loss: {:.3f}".format(epoch, training_loss, validation_loss))
            print("Epoch: {} | Training Accuracy: {:.3f} | Validation Accuracy: {:.3f} | Test Accuracy: {:.3f}".format(epoch, training_accuracy, validation_accuracy, test_accuracy))
            print("Epoch: {} | Training VizWiz Accuracy: {:.3f} | Validation VizWiz Accuracy: {:.3f} | Test VizWiz Accuracy: {:.3f}".format(epoch, training_vizwiz_accuracy, validation_vizwiz_accuracy, test_vizwiz_accuracy))
            print("Epoch: {} | Training Answerability Score: {:.3f} | Validation Answerability Score: {:.3f} | Test Answerability Score: {:.3f}\n".format(epoch, train_answerability_score, validation_answerability_score, test_answerability_score))

            #agrego esto para que solo me guarde el mejor modelo:
            best_validation_accuracy =  max(self.validation_accuracies)
            print('best_validation_accuracy: ', best_validation_accuracy)
            if save_path != None and epoch % save_every == 0:
            #if save_path != None and ((epoch % save_every == 0) and (validation_accuracy >= best_validation_accuracy)):
                self.save_model(save_path + "epoch_{}.pth".format(epoch))
        return

    def training_step(self, dataloader, criterion, optimizer, device):
        training_loss, training_accuracy, vizwiz_accuracy, total_sum = 0.0, 0.0, 0.0, 0
        answerable_true = []
        answerable_predicted = []
        self.train()
        print('training_step')
        for _, batch in tqdm(enumerate(dataloader)):
            #print('batch: ',len(batch))
            image, question, answer, answer_type, answers_for_questions, answerable = batch
            #print('image: ',image.size())
            #print('question. ',question.size())
            #print('answer: ', answer.size())
            #print('answer_type: ', answer_type.size())
            #print('answers_for_questions: ', answers_for_questions.size())
            #print('answerable: ', answerable.size())
            image, question, answer, answer_type, answers_for_questions, answerable = image.to(device), question.to(device), answer.to(device), answer_type.to(device), answers_for_questions.to(device), answerable.to(device)
            optimizer.zero_grad()
            output, answer_type_predicted, answerable_predict = self.forward(image, question)
            answerable = 1 - answerable
            answerable_predict = 1.0 - answerable_predict

            loss = criterion(output, answer) + criterion(answer_type_predicted, answer_type) + self.answerability_loss_fn(answerable_predict, answerable)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            predicted_answer = torch.argmax(output, dim = 1)
            actual_answer = torch.argmax(answer, dim = 1)
            for i in range(len(answer)):
                if actual_answer[i] == predicted_answer[i]:
                    training_accuracy +=1
                total_sum +=1
                vizwiz_accuracy += min(1, torch.sum(torch.eq(predicted_answer[i], answers_for_questions[i])).item()/3)
                answerable_true.append(answerable[i].item())
                answerable_predicted.append(answerable_predict[i].item())


        answerable_true = np.array(answerable_true)
        answerable_predicted = np.array(answerable_predicted)

        training_loss /= len(dataloader)
        training_accuracy /= total_sum
        vizwiz_accuracy /= total_sum

        return training_loss, training_accuracy, vizwiz_accuracy, average_precision_score(answerable_true, answerable_predicted, average = 'weighted')


    def validation_step(self, dataloader, criterion, device):
        validation_loss, validation_accuracy, vizwiz_accuracy, total_sum = 0.0, 0.0, 0.0, 0
        answerable_true = []
        answerable_predicted = []
        #print('validatiom_step')
        self.eval()
        with torch.no_grad():
            for _, batch in tqdm(enumerate(dataloader)):
                image, question, answer, answer_type, answers_for_questions, answerable = batch
                image, question, answer, answer_type, answers_for_questions, answerable = image.to(device), question.to(device), answer.to(device), answer_type.to(device), answers_for_questions.to(device), answerable.to(device)
                output, answer_type_predicted, answerable_predict = self.forward(image, question)

                # Answerablity is the confidence that quesion is not answerable, so we have to subtract from 1
                answerable = 1 - answerable
                answerable_predict = 1.0 - answerable_predict
                loss = criterion(output, answer) + criterion(answer_type_predicted, answer_type) + self.answerability_loss_fn(answerable_predict, answerable)
                validation_loss += loss.item()
                predicted_answer = torch.argmax(output, dim = 1)
                actual_answer = torch.argmax(answer, dim = 1)
                for i in range(len(answer)):
                    if torch.sum(answer[i]) == 0:
                        continue
                    if actual_answer[i] == predicted_answer[i]:
                        validation_accuracy += 1
                    total_sum +=1
                    vizwiz_accuracy += min(1, torch.sum(torch.eq(predicted_answer[i], answers_for_questions[i])).item()/3)
                    answerable_true.append(answerable[i].item())
                    answerable_predicted.append(answerable_predict[i].item())

        answerable_true = np.array(answerable_true)
        answerable_predicted = np.array(answerable_predicted)

        validation_loss /= len(dataloader)
        validation_accuracy /= total_sum
        vizwiz_accuracy /= total_sum

        # We will use weighted average since that there is imbalance in answerability in the dataset as displayed in EDA section
        return validation_loss, validation_accuracy, vizwiz_accuracy, average_precision_score(answerable_true, answerable_predicted, average = 'weighted')

    def test_step(self, dataloader):
        self.eval()
        accuracy, total_sum, vizwiz_accuracy = 0.0, 0, 0.0
        answerable_true = []
        answerable_predicted = []
        #print('test_step')
        with torch.no_grad():
            for _, batch in tqdm(enumerate(dataloader)):
                image, question, answer, answer_type, answers_for_questions, answerable = batch
                image, question, answer, answer_type, answers_for_questions, answerable = image.to(self.device), question.to(self.device), answer.to(self.device), answer_type.to(self.device), answers_for_questions.to(self.device), answerable.to(self.device)
                output, _, answerable_predict = self.forward(image, question)
                answerable = 1 - answerable
                answerable_predict = 1.0 - answerable_predict
                predicted_answer = torch.argmax(output, dim = 1)
                #predicted_answer = torch.argmax(output, dim = 0)
                actual_answer = torch.argmax(answer, dim = 1)
                for i in range(len(answer)):
                    if torch.sum(answer[i]) == 0:
                        continue
                    if predicted_answer[i] == actual_answer[i]:
                        accuracy += 1
                    vizwiz_accuracy += min(1, torch.sum(torch.eq(predicted_answer[i], answers_for_questions[i])).item()/3)
                    total_sum +=1
                    answerable_true.append(answerable[i].item())
                    answerable_predicted.append(answerable_predict[i].item())

        answerable_true = np.array(answerable_true)
        answerable_predicted = np.array(answerable_predicted)

        accuracy /= total_sum
        vizwiz_accuracy /= total_sum
        return accuracy, vizwiz_accuracy, average_precision_score(answerable_true, answerable_predicted, average = 'weighted')

    def save_model(self, path):
        """
        Saves the model state dictionary to the given path.

        Args:
        - self: the model object
        - path (str): the path to save the model state dictionary

        Returns:
        - None
        """
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """
        Loads the model state dictionary from the given path.

        Args:
        - self: the model object
        - path (str): the path to load the model state dictionary

        Returns:
        - self: the loaded model object
        """
        self.load_state_dict(torch.load(path))
        self.eval()
        return self

    def predict(self, image, question):
        """
        Predicts the output and answer type for the given image and question.

        Args:
        - self: the model object
        - image (tensor): the image tensor
        - question (tensor): the question tensor

        Returns:
        - output (tensor): the predicted output tensor
        - answer_type (str): the predicted answer type
        """
        output, answer_type, answerability = self.forward_testing(image, question)
        #self.forward(image, question)
        answerability = 1.0 - answerability
        return output, answer_type, answerability

    def plot_loss(self):
        """
        Plots the training and validation losses.

        Args:
        - self: the model object

        Returns:
        - None
        """
        plt.plot(self.training_losses, label = "Training Loss")
        plt.plot(self.validation_losses, label = "Validation Loss")
        plt.legend()
        plt.show()

    def plot_accuracy(self):
        """
        Plots the training and validation accuracies.

        Args:
        - self: the model object

        Returns:
        - None
        """
        plt.plot(self.training_accuracies, label = "Training Accuracy")
        plt.plot(self.validation_accuracies, label = "Validation Accuracy")
        plt.legend()
        plt.show()

    def plot_vizwiz_accuracy(self):
        """
        Plots the VizWiz training and validation accuracies.

        Args:
        - self: the model object

        Returns:
        - None
        """
        plt.plot(self.vizwiz_training_accuracies, label = "VizWiz Training Accuracy")
        plt.plot(self.vizwiz_validation_accuracies, label = "VizWiz Validation Accuracy")
        plt.legend()
        plt.show()

    def plot_answerability(self):
        """
        Plots the training and validation answerabilities.

        Args:
        - self: the model object

        Returns:
        - None
        """
        plt.plot(self.training_answerability, label = "Training Answerability")
        plt.plot(self.validation_answerability, label = "Validation Answerability")
        plt.legend()
        plt.show()
    def forward_testing(self, image, question):

        #print('image size: ', image.size())
        #print('question size: ',question.size())

        # Flattening and concatenating the image and question features
        image = torch.flatten(image, start_dim=0)
        #print(image)
        question = torch.flatten(question, start_dim=0)
        features = torch.cat((image, question), dim=0)

        # Calculating the answerability score
        answerability_score = self.answerability_linear_layer(features)
        answerability_score = self.answerability_final_layer(answerability_score)
        answerability_score = self.answerability_sigmoid(answerability_score)
        answerability_score = answerability_score.squeeze()

        # Passing the features through the first linear layer
        features = self.linear_layer1(features)
        #print('features: ', features)

        # Passing the features to get 4 answer types
        answer_type = self.answer_type_layer(features)

        # Expanding answer make to the same size as the number of classes (vocab size)
        answer_mask = self.answer_mask_layer(answer_type)

        # Applying sigmoid to get the answer mask
        answer_mask = self.sigmoid(answer_mask)

        # Passing the features through the second linear layer
        output = self.linear_layer2(features)

        # Applying the answer mask to the output
        output = output * answer_mask

        return output, answer_type, answerability_score


    def test_model(self, image_path, question):
        """
        Tests the model by predicting the answer and answer type for the given image and question.

        Args:
        - self: the model object
        - image_path (str): the path to the image file or URL
        - question (str): the question to be asked

        Returns:
        - predicted_answer (tensor): the predicted answer tensor
        - predicted_answer_type (str): the predicted answer type
        """
        #print('TEST MODEL')
        self.eval()
        if image_path.startswith("http"):
            image = Image.open(requests.get(image_path, stream = True).raw)
        else:
            image = Image.open(image_path)

        #image = self.preprocess(image).unsqueeze(0).to(self.device)
        #image_features = self.clip_model.encode_image(image)

        image_features = self.clip_img_model.encode(image,convert_to_tensor=True)   
        #print('image_features.shape ', image_features.shape)
        #question =  clip.tokenize(question).to(self.device)
        #question = clip.tokenize(question,truncate= True).to(self.device)
        #text_features = self.clip_model.encode_text(question).float()
        

        text_features = self.clip_text_model.encode([question], convert_to_tensor=True)
        #print('text_features.shape ', text_features.shape)

        predicted_answer, predicted_answer_type, answerability = self.predict(image_features, text_features)
        #print('predicted_answer:' , predicted_answer.shape,predicted_answer)
        return predicted_answer, predicted_answer_type, answerability

    def print_CLIP_model(self):
        """
        Prints the details of the selected CLIP model.

        Args:
        - self: the model object

        Returns:
        - None
        """
        input_resolution = self.clip_img_model.input_resolution
        context_length = self.clip_img_model.context_length
        vocab_size = self.clip_text_model.vocab_size

        print("Selected model:", self.model_name)
        print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in self.clip_model.parameters()]):,}")
        print("Input resolution:", input_resolution)
        print("Context length:", context_length)
        print("Vocab size:", vocab_size)
        print("")