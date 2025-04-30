from hart import get_hart_embeddings_one_doc_per_block
from hart import get_hart_embeddings as get_hart_embeddings_concat
from tqdm.auto import tqdm
import transformers
import pandas as pd
import numpy as np
import argparse
import torch
import os


def get_hart_embed_dir(args, suffix=''):   
    fdir, fpath = '/{}/bsize_{}'.format(args.mode, args.block_size), None
    if args.return_user_representation_as_mean_user_states:
        fpath = 'UserRepMeanUserStates{}.pt'.format(suffix)
    elif args.return_last_token_as_user_representation:
        fdir += '/{}'.format(args.representative_layer)
        if args.use_insep_as_last_token:
            fpath = 'UserRepLastInsepHS{}.pt'.format(suffix)
        else:
            fpath = 'UserRepLastWordHS{}.pt'.format(suffix)
    elif args.return_all_user_states:
        fpath = 'AllUserStates{}.pt'.format(suffix)
    elif args.return_word_embeddings:
        fdir += '/{}'.format(args.representative_layer)
        if args.return_word_embeds_with_insep:
            fpath = 'TokenEmbedWithInsep{}.pt'.format(suffix)
        else:
            fpath = 'TokenEmbedNoInsep{}.pt'.format(suffix)        
    elif args.return_document_embeddings:
        fdir += '/{}'.format(args.representative_layer)
        if args.use_insep_as_last_token:
            fpath = 'DocEmbedLastInsepHS{}.pt'.format(suffix)
        else:
            fpath = 'DocEmbedLastWordHS{}.pt'.format(suffix)
    else:
        print('Invalid State')
    fdir = 'embeddings/hart/{}'.format(args.dataset_name) + fdir
    return fdir, fpath


def generate_hart_embedding_handler(args):
    outputs = None
    if args.mode == 'concat':
        outputs = get_hart_embeddings_concat(
            model_path=args.model_path,
            data=args.data,
            text_column=args.text_column,
            user_id_column=args.user_id_column,
            text_id_column=args.text_id_column,
            block_size=args.block_size,
            max_blocks=args.max_blocks,
            batch_size=args.batch_size,
            order_by_column=args.order_by_column,
            retain_original_order=args.retain_original_order,
            return_document_embeddings=args.return_document_embeddings,
            return_user_representation_as_mean_user_states=args.return_user_representation_as_mean_user_states,
            return_last_token_as_user_representation=args.return_last_token_as_user_representation,
            use_insep_as_last_token=args.use_insep_as_last_token,
            return_all_user_states=args.return_all_user_states,
            return_output_with_data=args.return_output_with_data,
            return_word_embeddings=args.return_word_embeddings,
            return_word_embeds_with_insep=args.return_word_embeds_with_insep,
            representative_layer=args.representative_layer,
            return_pt=args.return_pt,
            is_a_test_case=args.is_a_test_case          
        )
    elif args.mode == 'odpb':
        outputs = get_hart_embeddings_one_doc_per_block(
            model_path=args.model_path,
            data=args.data,
            text_column=args.text_column,
            user_id_column=args.user_id_column,
            text_id_column=args.text_id_column,
            block_size=args.block_size,
            max_blocks=args.max_blocks,
            batch_size=args.batch_size,
            order_by_column=args.order_by_column,
            retain_original_order=args.retain_original_order,
            return_document_embeddings=args.return_document_embeddings,
            return_user_representation_as_mean_user_states=args.return_user_representation_as_mean_user_states,
            return_last_token_as_user_representation=args.return_last_token_as_user_representation,
            use_insep_as_last_token=args.use_insep_as_last_token,
            return_all_user_states=args.return_all_user_states,
            return_output_with_data=args.return_output_with_data,
            return_word_embeddings=args.return_word_embeddings,
            return_word_embeds_with_insep=args.return_word_embeds_with_insep,
            representative_layer=args.representative_layer,
            return_pt=args.return_pt,
            is_a_test_case=args.is_a_test_case          
        )

    fdir, fpath = get_hart_embed_dir(args, args.suffix)
    fdir = os.path.join(args.save_dir, fdir)
    print(fdir, fpath)
    os.makedirs(fdir, exist_ok=True)
    torch.save(outputs, os.path.join(fdir, fpath))


def get_other_model_and_tokenizer(model_name, device='cpu'):
    if model_name == 'gpt2':
        tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        model = transformers.AutoModel.from_pretrained('gpt2')
    elif model_name == 'bert':
        tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
        model = transformers.AutoModel.from_pretrained('bert-base-uncased')
    elif model_name == 'roberta':
        tokenizer = transformers.AutoTokenizer.from_pretrained('roberta-base')
        model = transformers.AutoModel.from_pretrained('roberta-base')
    else:
        raise ValueError('Invalid model name')
    device = torch.device(device)
    model.to(device)
    model.eval()
    return model, tokenizer


class MessageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample
    

def get_other_model_embeddings(model, tokenizer, data_loader, device, extract_cls):
    last_layer_embeds = []
    second_last_layer_embeds = []

    for batch in tqdm(data_loader):
        N = len(batch)
        encoded_input = tokenizer(batch, padding=True, return_tensors='pt')
        last_token_indices = encoded_input['attention_mask'].sum(dim=1) - 1
        with torch.no_grad():
            outputs = model(**(encoded_input.to(device)), output_hidden_states=True)
        last_layer_hidden_states = outputs.last_hidden_state.detach().cpu()
        second_last_layer_hidden_states = outputs.hidden_states[-2].detach().cpu()
        if extract_cls:
            last_layer_embeds.append(last_layer_hidden_states[:, 0])
            second_last_layer_embeds.append(second_last_layer_hidden_states[:, 0])
        else:
            last_layer_embeds.append(last_layer_hidden_states[torch.arange(N), last_token_indices])
            second_last_layer_embeds.append(second_last_layer_hidden_states[torch.arange(N), last_token_indices])
    last_layer_embeds = torch.vstack(last_layer_embeds).tolist()
    second_last_layer_embeds = torch.vstack(second_last_layer_embeds).tolist()
    last_layer_embeds_df = pd.DataFrame({'doc_embeds':last_layer_embeds})
    second_last_layer_embeds_df = pd.DataFrame({'doc_embeds':second_last_layer_embeds})

    return last_layer_embeds_df, second_last_layer_embeds_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='HaRT Embedding Script', description='Returns embeddings given a configuration')

    # Model
    parser.add_argument('--model_name', type=str, default='hart', choices=['concat', 'odpb'], help='Model name (hart, gpt2, bert, roberta)')
    
    # Common args for all models
    parser.add_argument('--data',  type=str, required=True, help='Path to pkl file')
    parser.add_argument('--batch_size',  type=int, default=5)
    parser.add_argument('--text_column',  type=str, default='all_essays')
    parser.add_argument('--save_dir', required=True, type=str)

    # Logistics
    parser.add_argument('--mode', required=True, choices=['concat', 'odpb'])
    parser.add_argument('--dataset_name',  type=str, default='dataset')
    parser.add_argument('--suffix',  type=str, default='', help='Can be used for versioning suffixes.')
    # parser.add_argument('--save_dir', required=True, type=str)
    parser.add_argument('--group_by_column',  type=str)

    # Arguments for get_hart_embeddings
    parser.add_argument('--model_path',  type=str, required=True, help='Path to HaRT model checkpoint')
    # parser.add_argument('--data',  type=str, required=True, help='Path to pkl file')
    # parser.add_argument('--text_column',  type=str, default='all_essays')
    parser.add_argument('--user_id_column',  type=str, default='user_id')
    parser.add_argument('--text_id_column',  type=str, default='message_id')
    parser.add_argument('--block_size',  type=int, required=True)
    parser.add_argument('--max_blocks',  type=int, default=None)
    # parser.add_argument('--batch_size',  type=int, default=5)
    parser.add_argument('--order_by_column',  type=str, default=None)
    parser.add_argument('--retain_original_order',  action='store_true')

    parser.add_argument('--return_document_embeddings',  action='store_true')
    parser.add_argument('--return_user_representation_as_mean_user_states',  action='store_true')
    parser.add_argument('--return_last_token_as_user_representation',  action='store_true')
    parser.add_argument('--use_insep_as_last_token',  action='store_true')
    parser.add_argument('--return_all_user_states',  action='store_true')
    parser.add_argument('--return_output_with_data',  action='store_true')
    parser.add_argument('--return_word_embeddings',  action='store_true')
    parser.add_argument('--return_word_embeds_with_insep',  action='store_true')
    parser.add_argument('--representative_layer', required=True, choices=['last', 'second_last'])
    parser.add_argument('--return_pt',  action='store_true')
    parser.add_argument('--is_a_test_case',  action='store_true')
    
    args = parser.parse_args()
    print(args)

    if args.model_name == 'hart':
        # Generate embeddings using HaRT
        generate_hart_embedding_handler(args)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        extract_cls = True if args.model_name in {'bert', 'roberta'} else False
        df = pd.read_pickle(args.data_path)
        test_dset = MessageDataset(df[args.text_col_name].values.tolist())
        test_loader = torch.utils.data.DataLoader(test_dset, batch_size=args.batch_size, shuffle=False)
        model, tokenizer = get_other_model_and_tokenizer(args.model_name, device)
        
        last_layer_embeds_df, second_last_layer_embeds_df = get_other_model_embeddings(model, tokenizer, test_loader, device, extract_cls)

        fdir = os.path.join(args.save_dir, 'embeddings', args.dataset_name, 'other_lm')
        os.makedirs(fdir, exist_ok=True)
        last_layer_embeds_df.to_pickle(os.path.join(fdir, '{}_last_layer_embeds.pkl'.format(args.model_name)))
        second_last_layer_embeds_df.to_pickle(os.path.join(fdir, '{}_second_last_layer_embeds.pkl'.format(args.model_name)))
