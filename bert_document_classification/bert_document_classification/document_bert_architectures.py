#from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertConfig, BertModel
from transformers import BertPreTrainedModel, BertConfig, BertModel
from transformers import DistilBertPreTrainedModel, DistilBertConfig, DistilBertModel

from torch import nn
import torch
from .transformer import TransformerEncoderLayer, TransformerEncoder

from torch.nn import LSTM


class DocumentBertLSTM(BertPreTrainedModel):
    """
    BERT output over document in LSTM
    """

    def __init__(self, bert_model_config: BertConfig):
        super(DocumentBertLSTM, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
        self.bert_batch_size= self.bert.config.bert_batch_size
        self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)
        self.lstm = LSTM(bert_model_config.hidden_size,bert_model_config.hidden_size, )
        self.classifier = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size, bert_model_config.num_labels),
            nn.Tanh()
        )

    #input_ids, token_type_ids, attention_masks
    def forward(self, document_batch: torch.Tensor, document_sequence_lengths: list, device='cuda'):

        #contains all BERT sequences
        #bert should output a (batch_size, num_sequences, bert_hidden_size)
        bert_output = torch.zeros(size=(document_batch.shape[0],
                                              min(document_batch.shape[1],self.bert_batch_size),
                                              self.bert.config.hidden_size), dtype=torch.float, device=device)

        #only pass through bert_batch_size numbers of inputs into bert.
        #this means that we are possibly cutting off the last part of documents.
        #use_grad = not freeze_bert
        #with torch.set_grad_enabled(False):
        
        for doc_id in range(document_batch.shape[0]):
            bert_output[doc_id][:self.bert_batch_size] = self.dropout(self.bert(document_batch[doc_id][:self.bert_batch_size,0],
                                            token_type_ids=document_batch[doc_id][:self.bert_batch_size,1],
                                            attention_mask=document_batch[doc_id][:self.bert_batch_size,2])[1])

        #lstm expects a ( num_sequences, batch_size (i.e. number of documents) , bert_hidden_size )
        #self.lstm.flatten_parameters()
        output, (_, _) = self.lstm(bert_output.permute(1,0,2))
        
        #print(bert_output.requires_grad)
        #print(output.requires_grad)

        last_layer = output[-1]
        #print("Last LSTM layer shape:",last_layer.shape)

        prediction = self.classifier(last_layer)
        #print("Prediction Shape", prediction.shape)
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction
    
    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True
                 
    def unfreeze_bert_encoder_last_layers(self):
        for name, param in self.bert.named_parameters():
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
                
    def unfreeze_bert_encoder_pooler_layer(self):
        for name, param in self.bert.named_parameters():
            if "pooler" in name:
                param.requires_grad = True
                         

class DistilBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class DocumentDistilBertLSTM( DistilBertPreTrainedModel ):
    """
    DistilBERT output over document in LSTM
    """

    def __init__(self, bert_model_config: DistilBertConfig ):
        super(DocumentDistilBertLSTM, self).__init__(bert_model_config)
        self.distilbert = DistilBertModel(bert_model_config)
        self.pooler=DistilBertPooler(bert_model_config)
        self.bert_batch_size= self.distilbert.config.bert_batch_size
        self.dropout = nn.Dropout(p=bert_model_config.dropout)
        self.lstm = LSTM(bert_model_config.hidden_size,bert_model_config.hidden_size, )
        self.classifier = nn.Sequential(
            nn.Dropout(p=bert_model_config.dropout),
            nn.Linear(bert_model_config.hidden_size, bert_model_config.num_labels),
            nn.Tanh()
        )
        self.init_weights()

    #input_ids, token_type_ids, attention_masks
    def forward(self, document_batch: torch.Tensor, document_sequence_lengths: list, device='cuda'):

        #contains all BERT sequences
        #bert should output a (batch_size (i.e. number of documents), num_sequences , bert_hidden_size)
        distilbert_output = torch.zeros(size=(document_batch.shape[0],
                                              min(document_batch.shape[1],self.bert_batch_size),
                                              self.distilbert.config.hidden_size), dtype=torch.float, device=device)
        
        #only pass through bert_batch_size numbers of inputs into bert.
        #this means that we are possibly cutting off the last part of documents.
    
        for doc_id in range(document_batch.shape[0]):
                
            hidden_states=self.distilbert(  input_ids=document_batch[doc_id][:self.bert_batch_size,0],
                                attention_mask=document_batch[doc_id][:self.bert_batch_size,2] )[0]
            #Output of distilbert is a tuple of length 1. First element (hidden_states) is of shape: 
            #( num_sequences(i.e. nr of sequences per document), nr_of_tokens(512) (i.e. nr of tokens per sequence), bert_hidden_size )
                        
            pooled_output=self.pooler( hidden_states )  # (num_sequences (i.e. nr of sequences per document), bert_hidden_size)
            
            distilbert_output[doc_id][:self.bert_batch_size]=self.dropout(pooled_output) #( #batch_size(i.e. number of documents) ,num_sequences (i.e. nr of sequences per document), bert_hidden_size)

        #lstm expects a ( num_sequences, batch_size (i.e. number of documents) , bert_hidden_size )
        self.lstm.flatten_parameters()
        output, (_, _) = self.lstm(distilbert_output.permute(1,0,2))
        
        last_layer = output[-1]

        prediction = self.classifier(last_layer)
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction
    
    def freeze_bert_encoder(self):
        for param in self.distilbert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.distilbert.parameters():
            param.requires_grad = True
            
    def unfreeze_bert_encoder_last_layers(self):
        for name, param in self.distilbert.named_parameters():
            if "layer.5" in name or "pooler" in name:
                param.requires_grad = True
                
    def unfreeze_bert_encoder_pooler_layer(self):
        for name, param in self.distilbert.named_parameters():
            if "pooler" in name:
                param.requires_grad = True         


class DocumentBertLinear(BertPreTrainedModel):
    """
    BERT output over document into linear layer
    """

    def __init__(self, bert_model_config: BertConfig):
        super(DocumentBertLinear, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
        self.bert_batch_size= self.bert.config.bert_batch_size
        self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)

        #self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6, norm=nn.LayerNorm(bert_model_config.hidden_size))
        self.classifier = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size * self.bert_batch_size, bert_model_config.num_labels),
            nn.Tanh()
        )

    #input_ids, token_type_ids, attention_masks
    def forward(self, document_batch: torch.Tensor, document_sequence_lengths: list, device='cuda'):

        #contains all BERT sequences
        #bert should output a (batch_size, num_sequences, bert_hidden_size)
        bert_output = torch.zeros(size=(document_batch.shape[0],
                                              min(document_batch.shape[1],self.bert_batch_size),
                                              self.bert.config.hidden_size), dtype=torch.float, device=device )

        #only pass through bert_batch_size numbers of inputs into bert.
        #this means that we are possibly cutting off the last part of documents.
        for doc_id in range(document_batch.shape[0]):
            bert_output[doc_id][:self.bert_batch_size] = self.dropout(self.bert(document_batch[doc_id][:self.bert_batch_size,0],
                                            token_type_ids=document_batch[doc_id][:self.bert_batch_size,1],
                                            attention_mask=document_batch[doc_id][:self.bert_batch_size,2])[1])

    
            
        prediction = self.classifier(bert_output.view(bert_output.shape[0], -1))
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction
    
    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True
                 
    def unfreeze_bert_encoder_last_layers(self):
        for name, param in self.bert.named_parameters():
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
                
    def unfreeze_bert_encoder_pooler_layer(self):
        for name, param in self.bert.named_parameters():
            if "pooler" in name:
                param.requires_grad = True
    

class DocumentBertMaxPool(BertPreTrainedModel):
    """
    BERT output over document into linear layer
    """

    def __init__(self, bert_model_config: BertConfig):
        super(DocumentBertMaxPool, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
        self.bert_batch_size= self.bert.config.bert_batch_size
        self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)

        # self.transformer_encoder = TransformerEncoderLayer(d_model=bert_model_config.hidden_size,
        #                                            nhead=6,
        #                                            dropout=bert_model_config.hidden_dropout_prob)
        #self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6, norm=nn.LayerNorm(bert_model_config.hidden_size))
        self.classifier = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size, bert_model_config.num_labels),
            nn.Tanh()
        )

    #input_ids, token_type_ids, attention_masks
    def forward(self, document_batch: torch.Tensor, document_sequence_lengths: list, device='cuda'):

        #contains all BERT sequences
        #bert should output a (batch_size, num_sequences, bert_hidden_size)
        bert_output = torch.zeros(size=(document_batch.shape[0],
                                              min(document_batch.shape[1],self.bert_batch_size),
                                              self.bert.config.hidden_size), dtype=torch.float, device=device)

        #only pass through bert_batch_size numbers of inputs into bert.
        #this means that we are possibly cutting off the last part of documents.
        for doc_id in range(document_batch.shape[0]):
            bert_output[doc_id][:self.bert_batch_size] = self.dropout(self.bert(document_batch[doc_id][:self.bert_batch_size,0],
                                            token_type_ids=document_batch[doc_id][:self.bert_batch_size,1],
                                            attention_mask=document_batch[doc_id][:self.bert_batch_size,2])[1])

        prediction = self.classifier(bert_output.max(dim=1)[0])
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction
    
    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True
                 
    def unfreeze_bert_encoder_last_layers(self):
        for name, param in self.bert.named_parameters():
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
                
    def unfreeze_bert_encoder_pooler_layer(self):
        for name, param in self.bert.named_parameters():
            if "pooler" in name:
                param.requires_grad = True
    
    
class DocumentBertTransformer(BertPreTrainedModel):
    """
    BERT -> TransformerEncoder -> Max over attention output.
    """

    def __init__(self, bert_model_config: BertConfig):
        super(DocumentBertTransformer, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
        self.bert_batch_size= self.bert.config.bert_batch_size
        self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)

        encoder_layer = TransformerEncoderLayer(d_model=bert_model_config.hidden_size,
                                                   nhead=6,
                                                   dropout=bert_model_config.hidden_dropout_prob)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6)
        self.classifier = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size, bert_model_config.num_labels),
            nn.Tanh()
        )

    #input_ids, token_type_ids, attention_masks
    def forward(self, document_batch: torch.Tensor, document_sequence_lengths: list, device='cuda'):

        #contains all BERT sequences
        #bert should output a (batch_size, num_sequences, bert_hidden_size)
        bert_output = torch.zeros(size=(document_batch.shape[0],
                                              min(document_batch.shape[1],self.bert_batch_size),
                                              self.bert.config.hidden_size), dtype=torch.float, device=device )

        #only pass through bert_batch_size numbers of inputs into bert.
        #this means that we are possibly cutting off the last part of documents.
        for doc_id in range(document_batch.shape[0]):
            bert_output[doc_id][:self.bert_batch_size] = self.dropout(self.bert(document_batch[doc_id][:self.bert_batch_size,0],
                                            token_type_ids=document_batch[doc_id][:self.bert_batch_size,1],
                                            attention_mask=document_batch[doc_id][:self.bert_batch_size,2])[1])

                
        transformer_output = self.transformer_encoder(bert_output.permute(1,0,2))

        #print(transformer_output.shape)

        prediction = self.classifier(transformer_output.permute(1,0,2).max(dim=1)[0])
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction
    
    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True
                 
    def unfreeze_bert_encoder_last_layers(self):
        for name, param in self.bert.named_parameters():
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
                
    def unfreeze_bert_encoder_pooler_layer(self):
        for name, param in self.bert.named_parameters():
            if "pooler" in name:
                param.requires_grad = True
    