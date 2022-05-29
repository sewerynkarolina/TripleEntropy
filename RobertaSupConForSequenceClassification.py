import torch
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaClassificationHead
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from pytorch_metric_learning import losses


class RobertaSupConForSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        self.init_weights()

        self.temperature = kwargs.get('temperature', None)
        self.beta = kwargs.get('beta', None)

        self.supcon = losses.SupConLoss(
                    num_classes=self.num_labels,
                    embedding_size=config.hidden_size,
                    temperature=self.temperature)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            ce_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            populated_labels = torch.ones(sequence_output.shape[0] * sequence_output.shape[1]).type(torch.LongTensor).to(ce_loss.device)

            counter = 0
            for label in labels.view(-1):
                for idx in range(sequence_output.shape[1]):
                    populated_labels[counter] = label
                    counter += 1
            supcon_loss = self.supcon(sequence_output.view(-1, sequence_output.shape[2]), populated_labels)
            loss = self.beta * ce_loss + (1 - self.beta) * (supcon_loss)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
