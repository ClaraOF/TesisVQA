import datetime
import numpy as np
import torch
import pandas as pd
from os import path
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(__file__))  # agrega carpeta actual al path
from metrics import compute_metrics

# IMPLEMENTAR ACÁ EL TRAINING LOOP
def training_loop(start_epoch,n_epochs, scheduler, optimizer, model, train_loader, validation_loader,device, exp_name, best_model_chekpoint, new_datasets_folder, metrics_log, answer_space):
  init = datetime.datetime.today()
  print('Init time: ', init)
  # inicializamos una lista de valores de loss por época
  training_loss_per_epoch = []
  validation_loss_per_epoch = []
  # e igual del accuracy
  training_accuracy_per_epoch = []
  validation_accuracy_per_epoch = []
  training_metrics_per_epoch =[]
  validation_metrics_per_epoch =[]

  # iteramos por cada época
  for epoch in range(start_epoch, n_epochs + 1):
    print('epoch: ',epoch)
    init_epoch = datetime.datetime.today()
    training_metrics_per_batch=[]
    validation_metrics_per_batch =[]
    # -----------------------------
    #          TRAINING
    # -----------------------------

    # inicializamos el valor de la loss en 0
    loss_train = 0.0
    # y la cantidad de valores totales y correctos (para calcular accuracy)
    total = 0
    correct = 0
    model.train()
    # for cada batch en nuestros datos de entrenamiento
    print('Training...')

    for batch in tqdm(train_loader):

        # get the inputs;
        batch = {k:v.to(device) for k,v in batch.items()}

        #me quedo con los labels para poder comparar mas adelante
        #labels = torch.argmax(batch["labels"],dim=-1)
        #labels = torch.squeeze(labels)
        #labels = labels.to(device=device)
        labels = batch['labels']
        #print('labels shape: ', labels.shape[0])

        # zero the parameter gradients
        #optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(**batch)

        logits = outputs["logits"] # training outputs
        #evaluamos la loss
        loss = outputs["loss"] #outputs.loss

        # limpiamos los gradientes anteriores
        optimizer.zero_grad()
        # calculamos los gradientes con backpropagation
        loss.backward()
        # actualizamos el optimizador
        optimizer.step()

        # y vamos sumando los valores de la loss
        loss_train += loss.item()

        # también podemos calcular el accuracy sobre datos de entrenamiento:
        # primero tenemos que estimar las etiquetas
        #_, predicted = torch.max(model.final_activation(outputs), dim=1)
        predictions =  torch.argmax(logits,dim=-1)
        #print('predictions: ', predictions)
        #print('labels: ', labels)

        # y después calculamos total de muestras y correctas
        #print('labels: ', batch["labels"].size()[0])
        total += labels.shape[0]
        correct += int((predictions == labels).sum())
        #train_tuple=(logits.argmax(axis=-1).cpu(), labels)
        #training_metrics_per_batch.append(compute_metrics(train_tuple))
        #print('correct: ', correct)
    # calculamos el promedio de la loss por época
    training_loss_per_epoch.append(loss_train / len(train_loader))
    # y el valor de accuracy
    training_accuracy_per_epoch.append(correct / total)
    #training_metrics_per_epoch.append(training_metrics_per_batch)

    # -----------------------------
    #         VALIDATION
    # -----------------------------

    # inicializamos el valor de la loss en 0
    loss_val = 0.0
    # y la cantidad de valores totales y correctos (para calcular accuracy)
    total = 0
    correct = 0
    model.eval()

    # como iteramos sobre los datos de validación, no calculamos gradientes
    with torch.no_grad():
      # for cada batch en nuestros datos de validación
      print('Validation...')
      for batch in tqdm(validation_loader):
        # get the inputs;
        batch = {k: v.to(device) for k,v in batch.items()}
        #me quedo con los labels
        #labels = torch.argmax(batch["labels"],dim=-1)
        #labels = torch.squeeze(labels)
        #labels = labels.to(device=device)
        labels = batch['labels']

        # obtenemos la rta del modelo
        outputs = model(**batch)
        logits = outputs["logits"]  #outputs.logits
        # evaluamos la loss
        loss = outputs["loss"] #outputs.loss

        #print("Loss:", loss.item())
        # y vamos sumando sus valores
        loss_val += loss.item()

        # estimamos las etiquetas
        predictions =  torch.argmax(logits,dim=-1)
        # y calculamos total de muestras y correctas
        total += labels.shape[0]
        correct += int((predictions == labels).sum())
        #val_tuple=(logits.argmax(axis=-1).cpu(), labels)
        #validation_metrics_per_batch.append(compute_metrics(val_tuple))
        #####################################
        #Calculo otra metrica que no sea ACC entre los datos predichos y los labels
        #####################################
        #primero tengo que convertir los tensoes en array de numpay para poder aplicarle las funciones de metricas:
        val_tuple=(predictions.cpu().detach().numpy(), labels.cpu().detach().numpy())
        validation_metrics_per_batch.append(compute_metrics(val_tuple,answer_space))


    # calculamos el promedio de la loss por época
    validation_loss_per_epoch.append(loss_val / len(validation_loader))
    # y el valor de accuracy
    validation_accuracy_per_epoch.append(correct / total)

    #caculate mean metrics by batch to save it in each epoch:
    cosine_scores_per_batch = np.mean([validation_metrics_per_batch[i]['cosine'] for i, val in enumerate(validation_metrics_per_batch)])
    wups_scores_per_batch=np.mean([validation_metrics_per_batch[i]['wups'] for i, val in enumerate(validation_metrics_per_batch)])
    validation_metrics_per_batch ={'cosine': cosine_scores_per_batch, 'wups': wups_scores_per_batch}

    validation_metrics_per_epoch.append(validation_metrics_per_batch)

    ####################################
    #agrego esta linea para ir disminuyendo el lr
    curr_lr = optimizer.param_groups[0]['lr']
    print(f'LR: {curr_lr}')
    # Note that step should be called after validate()
    scheduler.step(loss_val) #->lo comento cuando quiero usar un LR fijo
    ####################################

    # -----------------------------
    #    EARLY STOPPING
    # -----------------------------
    """
    Mas que early stopping es una implenetacion de almacenamiento de 2 modelos:
        * el ultimo
        * el de mejor accuracy en validacion
    """

    # guardamos el mejor modelo

    best_validation_accuracy =  max(validation_accuracy_per_epoch)

    if validation_accuracy_per_epoch[-1] == best_validation_accuracy:
        torch.save(model.state_dict(), path.join(new_datasets_folder, best_model_chekpoint))

    # guardamos el ultimo modelo
    #para coder retomar un entrenamiento desde un checkpoint tengo que guardar un par de cosas mas
    '''
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimazer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'curr_lr': curr_lr
                },path.join(new_datasets_folder, f'last_VQA_Multimodel_VIZWIZ_finetunning_in_loop_ES.pt'))
    #torch.save(model.state_dict(), path.join(new_datasets_folder, f'last_ViLT_VIZWIZ_finetunning_in_loop.pt'))
    '''
    print('Best models saved')

    # -----------------------------
    #    IMPRIMIMOS RESULTADOS
    # -----------------------------

    text = f'[Epoch: {epoch}]:\t'
    text += f'- Training loss: {training_loss_per_epoch[-1]:.4f}.'
    text += f'- Validation loss: {validation_loss_per_epoch[-1]:.4f}'
    text += f'- Training accuracy: {training_accuracy_per_epoch[-1]}'
    text += f'- Validation accuracy: {validation_accuracy_per_epoch[-1]:.4f}'
    text +=f'- Validation WUP: {validation_metrics_per_epoch[-1]}'


    # cada 10 épocas, mostramos cuánto tardó la época y qué valor obtuvimos
    if epoch == 1 or epoch % 5 == 0:
        print(text+"\n")

    end_epoch = datetime.datetime.today()

    with open(path.join(new_datasets_folder, "log.log"), "a+") as log_file:
        log_file.write(f"[{end_epoch}]  {text}. Elapsed {(end_epoch - init_epoch).seconds / 60 :.1f} minutes.\n")

    #guardo las loss y accuracy en un df asi despues lo puedo usar para graficar
    loss_acc_data =[{'time': end_epoch,
                     'epoch': epoch,
                     'tr_loss':round(training_loss_per_epoch[-1],4),
                     'val_loss':round(validation_loss_per_epoch[-1],4),
                     'tr_acc':round(training_accuracy_per_epoch[-1],4),
                     'val_acc':round(validation_accuracy_per_epoch[-1],4),
                     'val_wups':round(validation_metrics_per_epoch[-1]['wups'],4),
                     'val_cosine':round(validation_metrics_per_epoch[-1]['cosine'],4),
                     'curr_lr': str(curr_lr),'exp_name': exp_name}] #,'tr_metrics':round(training_metrics_per_epoch[-1],4),'val_metrics':round(validation_metrics_per_epoch[-1],4)}]
    loss_acc_df = pd.DataFrame(loss_acc_data)

    # append data frame to CSV file
    loss_acc_df.to_csv(new_datasets_folder + metrics_log, mode='a', index=False, header=False)


  end = datetime.datetime.today()
  with open(path.join(new_datasets_folder, "log.log"), "a+") as log_file:
    log_file.write(f"[{end_epoch}] FIN DEL TRAINING LOOP. Mejor modelo hallado en iteracion {np.argmax(validation_accuracy_per_epoch)+1}. Total elapsed {(end-init).seconds/60:.1f} minutos\n{'='*25}")


    # cada 10 épocas, mostramos cuánto tardó la época y qué valor obtuvimos
    if epoch == 1 or epoch % 10 == 0:
      print('Epoch {}:'.format(epoch))
      print(' ---> Loss: Training {:.4f} - Validation {:.4f}'.format(training_loss_per_epoch[-1], validation_loss_per_epoch[-1]))
      print(' ---> Accuracy: Training {:.4f} - Validation {:.4f}'.format(training_accuracy_per_epoch[-1], validation_accuracy_per_epoch[-1]))

  # devolvemos los resultados
  return model, training_loss_per_epoch, validation_loss_per_epoch, training_accuracy_per_epoch, validation_accuracy_per_epoch

# def training_loop(start_epoch, n_epochs, scheduler, optimizer, model, train_loader, validation_loader, device, exp_name, best_model_chekpoint, new_datasets_folder, metrics_log, answer_space):
#     training_loss_per_epoch = []
#     validation_loss_per_epoch = []
#     training_accuracy_per_epoch = []
#     validation_accuracy_per_epoch = []
#     validation_metrics_per_epoch = []

#     for epoch in range(start_epoch, n_epochs + 1):
#         init_epoch = datetime.datetime.today()
#         model.train()
#         loss_train = 0.0
#         total = 0
#         correct = 0
#         for batch in tqdm(train_loader):
#             batch = {k: v.to(device) for k, v in batch.items()}
#             labels = batch['labels']
#             outputs = model(**batch)
#             logits = outputs["logits"]
#             loss = outputs["loss"]
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             loss_train += loss.item()
#             predictions = torch.argmax(logits, dim=-1)
#             total += labels.shape[0]
#             correct += int((predictions == labels).sum())
#         training_loss_per_epoch.append(loss_train / len(train_loader))
#         training_accuracy_per_epoch.append(correct / total)

#         model.eval()
#         loss_val = 0.0
#         total = 0
#         correct = 0
#         validation_metrics_per_batch = []
#         with torch.no_grad():
#             for batch in tqdm(validation_loader):
#                 batch = {k: v.to(device) for k, v in batch.items()}
#                 labels = batch['labels']
#                 outputs = model(**batch)
#                 logits = outputs["logits"]
#                 loss = outputs["loss"]
#                 loss_val += loss.item()
#                 predictions = torch.argmax(logits, dim=-1)
#                 total += labels.shape[0]
#                 correct += int((predictions == labels).sum())
#                 val_tuple = (predictions.cpu().detach().numpy(), labels.cpu().detach().numpy())
#                 validation_metrics_per_batch.append(compute_metrics(val_tuple, answer_space))
#         validation_loss_per_epoch.append(loss_val / len(validation_loader))
#         validation_accuracy_per_epoch.append(correct / total)
#         cosine_scores_per_batch = np.mean([m['cosine'] for m in validation_metrics_per_batch])
#         wups_scores_per_batch = np.mean([m['wups'] for m in validation_metrics_per_batch])
#         validation_metrics_per_epoch.append({'cosine': cosine_scores_per_batch, 'wups': wups_scores_per_batch})

#         curr_lr = optimizer.param_groups[0]['lr']
#         scheduler.step(loss_val)
#         best_validation_accuracy = max(validation_accuracy_per_epoch)
#         if validation_accuracy_per_epoch[-1] == best_validation_accuracy:
#             torch.save(model.state_dict(), path.join(new_datasets_folder, best_model_chekpoint))
#         print('Best models saved')

#         # -----------------------------
#         #    IMPRIMIMOS RESULTADOS
#         # -----------------------------

#         text = f'[Epoch: {epoch}]:\t'
#         text += f'- Training loss: {training_loss_per_epoch[-1]:.4f}.'
#         text += f'- Validation loss: {validation_loss_per_epoch[-1]:.4f}'
#         text += f'- Training accuracy: {training_accuracy_per_epoch[-1]}'
#         text += f'- Validation accuracy: {validation_accuracy_per_epoch[-1]:.4f}'
#         text +=f'- Validation WUP: {validation_metrics_per_epoch[-1]}'

#             # cada 10 épocas, mostramos cuánto tardó la época y qué valor obtuvimos
#         if epoch == 1 or epoch % 5 == 0:
#             print(text+"\n")
#         end_epoch = datetime.datetime.today()
#         with open(path.join(new_datasets_folder, "log.log"), "a+") as log_file:
#             log_file.write(f"[{end_epoch}]  {text}. Elapsed {(end_epoch - init_epoch).seconds / 60 :.1f} minutes.\n")
        
#         loss_acc_data = [{
#             'time': end_epoch,
#             'epoch': epoch,
#             'tr_loss': round(training_loss_per_epoch[-1], 4),
#             'val_loss': round(validation_loss_per_epoch[-1], 4),
#             'tr_acc': round(training_accuracy_per_epoch[-1], 4),
#             'val_acc': round(validation_accuracy_per_epoch[-1], 4),
#             'val_wups': round(validation_metrics_per_epoch[-1]['wups'], 4),
#             'val_cosine': round(validation_metrics_per_epoch[-1]['cosine'], 4),
#             'curr_lr': str(curr_lr),
#             'exp_name': exp_name
#         }]
#         loss_acc_df = pd.DataFrame(loss_acc_data)
#         loss_acc_df.to_csv(path.join(new_datasets_folder, metrics_log), mode='a', index=False, header=False)
#         # cada 10 épocas, mostramos cuánto tardó la época y qué valor obtuvimos
#         if epoch == 1 or epoch % 10 == 0:
#             print('Epoch {}:'.format(epoch))
#             print(' ---> Loss: Training {:.4f} - Validation {:.4f}'.format(training_loss_per_epoch[-1], validation_loss_per_epoch[-1]))
#             print(' ---> Accuracy: Training {:.4f} - Validation {:.4f}'.format(training_accuracy_per_epoch[-1], validation_accuracy_per_epoch[-1]))

#     return model, training_loss_per_epoch, validation_loss_per_epoch, training_accuracy_per_epoch, validation_accuracy_per_epoch