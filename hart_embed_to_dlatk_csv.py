from generate_embeddings import get_hart_embed_dir
import pandas as pd
import numpy as np
import argparse
import torch
import os


def get_dlatk_df(embeds, embed_column='user_reps', id_column='user_id'):
    all_df_rows = []
    for i in range(embeds.shape[0]):
        embed = embeds.iloc[i][embed_column]
        embed_id = embeds.iloc[i][id_column]
        df_rows = []
        for i in range(len(embed)):
            df_rows.append(
                {
                    'group_id': embed_id,
                    'feat': '{}me'.format(i),
                    'value': embed[i],
                    'group_norm': embed[i]
                }
            )
        all_df_rows.extend(df_rows)
    dlatk_df = pd.DataFrame(all_df_rows)
    return dlatk_df


def load_user_embeds(fpath, user_id_column='user_id'):
    embeds = torch.load(fpath)['user_representations']
    embeds['user_reps'] = embeds['user_reps'].apply(lambda x: x.numpy())
    user_embeds = pd.DataFrame()
    user_embeds[[user_id_column, 'embedding']] = embeds[[user_id_column, 'user_reps']]
    return user_embeds


def load_doc_embeds(fpath, text_id_column='message_id'):
    embeds = torch.load(fpath)['document_embeddings']
    embeds['doc_embeds'] = embeds['doc_embeds'].apply(lambda x: x.numpy())
    doc_embeds = pd.DataFrame()
    doc_embeds[[text_id_column, 'embedding']] = embeds[[text_id_column, 'doc_embeds']]
    return doc_embeds


def load_token_embeds(fpath, text_id_column='message_id'):
    embeds = torch.load(fpath)['word_embeddings']
    embeds['token_embeds'] = embeds['token_embeds'].apply(lambda x: x.numpy())
    token_embeds = pd.DataFrame()
    token_embeds[[text_id_column, 'embedding']] = embeds[[text_id_column, 'token_embeds']]
    return token_embeds


def get_doc_embeds_from_tokens(token_embeds, text_id_column='message_id'):
    # Average token embeddings in a document to get document-level embedding
    doc_embeds = pd.DataFrame()
    doc_embeds[text_id_column] = token_embeds[text_id_column]
    doc_embeds['embedding'] = token_embeds['embedding'].apply(lambda x: x.mean(axis=0))
    return doc_embeds


def group_doc_embeds_by_column(doc_embeds, input_df_path, group_by_column='user_id', text_id_column='message_id'):
    inputdf = pd.read_pickle(input_df_path)
    doc_embeds_df = inputdf[[group_by_column, text_id_column]].join(doc_embeds.set_index(text_id_column), on=text_id_column, how='inner')
    # doc_embeds_df[group_by_column] = doc_embeds_df[group_by_column].astype(int)
    agg_embeds = doc_embeds_df.groupby(group_by_column, sort=False).agg({'embedding': list}).reset_index()
    agg_embeds['embedding'] = agg_embeds['embedding'].apply(lambda x: np.mean(x, axis=0))
    return agg_embeds


def get_csv_dir(args, suffix=''):   
    fdir, fpath = '/{}/bsize_{}'.format(args.mode, args.block_size), None
    if args.return_user_representation_as_mean_user_states:
        fpath = 'UserRepMeanUserStates{}.csv'.format(suffix)
    elif args.return_last_token_as_user_representation:
        fdir += '/{}'.format(args.representative_layer)
        if args.use_insep_as_last_token:
            fpath = 'UserRepLastInsepHS{}.csv'.format(suffix)
        else:
            fpath = 'UserRepLastWordHS{}.csv'.format(suffix)
    elif args.return_all_user_states:
        fpath = 'AllUserStates{}.pt'.format(suffix)
    elif args.return_word_embeddings:
        fdir += '/{}'.format(args.representative_layer)
        if args.return_word_embeds_with_insep:
            fpath = 'TokenEmbedWithInsep{}.csv'.format(suffix)
        else:
            fpath = 'TokenEmbedNoInsep{}.csv'.format(suffix)
        
        if args.group_by_column != None:
            fpath = args.group_by_column + '.' + fpath
        else:
            fpath = args.text_id_column + '.' + fpath
    elif args.return_document_embeddings:
        fdir += '/{}'.format(args.representative_layer)
        if args.use_insep_as_last_token:
            fpath = 'DocEmbedLastInsepHS{}.csv'.format(suffix)
        else:
            fpath = 'DocEmbedLastWordHS{}.csv'.format(suffix)
        
        if args.group_by_column != None:
            fpath = args.group_by_column + '.' + fpath
        else:
            fpath = args.text_id_column + '.' + fpath
    else:
        print('Invalid State')
    fdir = 'tables/{}'.format(args.dataset_name) + fdir
    return fdir, fpath



def generate_csv_handler(args):
    fdir, fname = get_hart_embed_dir(args, args.suffix)
    fpath = os.path.join(args.save_dir, fdir, fname)

    if args.return_user_representation_as_mean_user_states or args.return_last_token_as_user_representation:
        # load and directly convert to df
        user_embeds = load_user_embeds(fpath, user_id_column=args.user_id_column)
        df = get_dlatk_df(user_embeds, embed_column='embedding', id_column=args.user_id_column)
    elif args.return_all_user_states:
        print("Coming soon")
    elif args.return_word_embeddings:
        token_embeds = load_token_embeds(fpath)
        # get doc level embed by averaging tokens
        doc_embeds = get_doc_embeds_from_tokens(token_embeds, text_id_column=args.text_id_column)
        if args.group_by_column != None:
            # if group_by_column is provided, then aggregate on this column and create the df from resulting embeds
            averaged_embeds = group_doc_embeds_by_column(doc_embeds, args.data, group_by_column=args.group_by_column, text_id_column=args.text_id_column)
            df = get_dlatk_df(averaged_embeds, embed_column='embedding', id_column=args.group_by_column)
        else:
            # no group_by_column means user wants to save doc embeddings at doc-level itself.
            df = get_dlatk_df(doc_embeds, embed_column='embedding', id_column=args.text_id_column)
    elif args.return_document_embeddings:
        doc_embeds = load_doc_embeds(fpath)
        if args.group_by_column != None:
            averaged_embeds = group_doc_embeds_by_column(doc_embeds, args.data, group_by_column=args.group_by_column, text_id_column=args.text_id_column)
            df = get_dlatk_df(averaged_embeds, embed_column='embedding', id_column=args.group_by_column)
        else:
            df = get_dlatk_df(doc_embeds, embed_column='embedding', id_column=args.text_id_column)
    else:
        print('Invalid State')

    fdir, fname = get_csv_dir(args, args.suffix)
    fdir = os.path.join(args.save_dir, fdir)
    print(fdir, fname)
    os.makedirs(fdir, exist_ok=True)
    df.to_csv(os.path.join(fdir, fname))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='HaRT Embed -> DLATK CSV Script', description='Returns CSV given a configuration')

    # Additional
    parser.add_argument('--mode', required=True, choices=['concat', 'odpb'])
    parser.add_argument('--dataset_name',  type=str, default='dataset')
    parser.add_argument('--suffix',  type=str, default='', help='Can be used for versioning suffixes.')
    parser.add_argument('--save_dir', required=True, type=str)
    parser.add_argument('--group_by_column',  type=str)

    # Arguments for get_hart_embeddings
    parser.add_argument('--model_path',  type=str, required=True, help='Path to HaRT model checkpoint')
    parser.add_argument('--data',  type=str, required=True, help='Path to pkl file')
    parser.add_argument('--text_column',  type=str, default='all_essays')
    parser.add_argument('--user_id_column',  type=str, default='user_id')
    parser.add_argument('--text_id_column',  type=str, default='message_id')
    parser.add_argument('--block_size',  type=int, required=True)
    parser.add_argument('--max_blocks',  type=int, default=None)
    parser.add_argument('--batch_size',  type=int, default=5)
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

    generate_csv_handler(args)

