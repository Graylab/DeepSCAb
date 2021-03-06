import argparse
import torch
import os
import torch.nn as nn
import torch.utils.data as data
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import deepscab
from deepscab.models.AbChiResNet.AbChiResNet import AbChiResNet
from deepscab.util.util import RawTextArgumentDefaultsHelpFormatter
from deepscab.datasets.H5AbChiDataset import H5AbChiDataset
from deepscab.preprocess.create_antibody_db import download_train_dataset
from deepscab.preprocess.generate_h5_AbChi_files import antibody_to_h5

_output_names = [
    'dist', 'omega', 'theta', 'phi', 'chi_one', 'chi_two', 'chi_three',
    'chi_four', 'chi_five'
]


def train(model,
          train_loader,
          validation_loader,
          optimizer,
          epochs,
          device,
          criterion,
          lr_modifier,
          writer,
          save_file,
          properties=None):
    """"""
    properties = {} if properties is None else properties
    print('Using {} as device'.format(str(device).upper()))
    model = model.to(device)

    save_every = 5
    for epoch in range(epochs):
        out_weights = [1, 1, 1, 1, 0.001, 0.001, 0.001, 0.001, 0.001]
        if epoch > 0:
            out_weights = [1, 1, 1, 1, 0.1, 0.05, 0.05, 0.05, 0.001]
        if epoch > 10:
            out_weights = [1, 2, 2, 1, 0.5, 0.4, 0.3, 0.2, 0.1]
        if epoch > 20:
            out_weights = [1, 5, 5, 1, 5, 2.5, 1, 1, 0.1]
        if epoch > 30:
            out_weights = [1, 2, 2, 1, 0.1, 0.05, 0.05, 0.05, 0.001]

        train_losses = _train_epoch(model, train_loader, optimizer, device,
                                    criterion)

        avg_train_losses = train_losses / len(train_loader)
        train_loss_dict = dict(
            zip(_output_names + ['total'], avg_train_losses.tolist()))
        writer.add_scalars('train_loss', train_loss_dict, global_step=epoch)
        print('\nAverage training loss (epoch {}): {}'.format(
            epoch, train_loss_dict))

        val_losses = _validate(model, validation_loader, device, criterion)
        avg_val_losses = val_losses / len(validation_loader)
        val_loss_dict = dict(
            zip(_output_names + ['total'], avg_val_losses.tolist()))
        writer.add_scalars('validation_loss', val_loss_dict, global_step=epoch)
        print('\nAverage validation loss (epoch {}): {}'.format(
            epoch, avg_val_losses.tolist()))

        total_val_loss = val_losses[-1]
        lr_modifier.step(total_val_loss)

        if (epoch + 1) % save_every == 0:
            properties.update({'model_state_dict': model.state_dict()})
            properties.update({
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss_dict,
                'val_loss': val_loss_dict,
                'epoch': epoch
            })
            torch.save(properties, save_file + ".e{}".format(epoch + 1))

    properties.update({'model_state_dict': model.state_dict()})
    properties.update({
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss_dict,
        'val_loss': val_loss_dict,
        'epoch': epoch
    })
    torch.save(properties, save_file)


def _train_epoch(model, train_loader, optimizer, device, criterion):
    """Trains a model for one epoch"""
    model.train()
    running_losses = torch.zeros(10)
    for inputs, labels in tqdm(train_loader, total=len(train_loader)):
        inputs = inputs.to(device)
        labels = [label.to(device) for label in labels]

        optimizer.zero_grad()

        def handle_batch():
            """Function done to ensure variables immediately get dealloced"""
            outputs = model(inputs)
            losses = [
                criterion(output, label)
                for output, label in zip(outputs, labels)
            ]
            total_loss = sum(losses)
            losses.append(total_loss)

            total_loss.backward()
            optimizer.step()
            return outputs, torch.Tensor([float(l.item()) for l in losses])

        outputs, batch_loss = handle_batch()
        running_losses += batch_loss

    return running_losses


def _validate(model, validation_loader, device, criterion):
    """"""
    with torch.no_grad():
        model.eval()
        running_losses = torch.zeros(10)
        for inputs, labels in tqdm(validation_loader,
                                   total=len(validation_loader)):
            inputs = inputs.to(device)
            labels = [label.to(device) for label in labels]

            def handle_batch():
                """Function done to ensure variables immediately get dealloced"""
                outputs = model(inputs)
                losses = [
                    criterion(output, label)
                    for output, label in zip(outputs, labels)
                ]
                total_loss = sum(losses)
                losses.append(total_loss)

                return outputs, torch.Tensor([float(l.item()) for l in losses])

            outputs, batch_loss = handle_batch()
            running_losses += batch_loss
    return running_losses


def _get_args():
    """Gets command line arguments"""
    project_path = os.path.abspath(os.path.join(deepscab.__file__, "../.."))

    desc = ('''
        Script for training a model using a non-redundant set of bound and 
        unbound antibodies from SabDab with at most 99% sequence similarity, 
        a resolution cutoff of 3, and with a paired VH/VL. By default, uses 
        the model from https://doi.org/10.1101/2020.02.09.940254.\n
        \n
        If there is no H5 file named antibody.h5 in the deepscab/data directory, 
        then the script automatically uses the PDB files in 
        deepscab/data/antibody_database directory to generate antibody.h5. If no 
        such directory exists, then the script downloads the set of pdbs from
        SabDab outlined above.
        ''')
    parser = argparse.ArgumentParser(
        description=desc, formatter_class=RawTextArgumentDefaultsHelpFormatter)
    # Model architecture arguments
    parser.add_argument('--num_blocks1D',
                        type=int,
                        default=3,
                        help='Number of one-dimensional ResNet blocks to use.')
    parser.add_argument('--num_blocks2D',
                        type=int,
                        default=25,
                        help='Number of two-dimensional ResNet blocks to use.')
    parser.add_argument('--dilation_cycle', type=int, default=5)
    parser.add_argument(
        '--num_bins',
        type=int,
        default=36,
        help=('Number of bins to discretize the continuous '
              'distance, and angle values into.\n'
              'Example:\n'
              'For residue pairs i and j, let d be the euclidean '
              'distance between them in Angstroms. A num_bins of '
              '36 would discretize each d into the following bins:\n'
              '[(0, 0.5), (0.5, 1.0), (1.0, 1.5)... (17.5, Inf)]\n'
              'Depending on the type of inter-residue angle, '
              'angles are binned into 36 evenly spaced bins between '
              '0 (or -180) and 180 degrees'))
    parser.add_argument('--dropout',
                        type=float,
                        default=0.2,
                        help=('The chance of entire channels being zeroed out '
                              'during training'))

    # Training arguments
    parser.add_argument('--epochs',
                        type=int,
                        default=50,
                        help='Number of epochs')
    parser.add_argument('--save_every',
                        type=int,
                        default=1,
                        help='Save model every X number of epochs.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=4,
                        help='Number of proteins per batch')
    parser.add_argument('--lr',
                        type=float,
                        default=0.01,
                        help='Learning rate for Adam')
    parser.add_argument('--try_gpu',
                        type=bool,
                        default=True,
                        help='Whether or not to check for/use the GPU')
    parser.add_argument('--train_split',
                        type=float,
                        default=0.95,
                        help=('The percentage of the dataset that is used '
                              'during training'))

    default_h5_file = os.path.join(project_path, 'data/abAbChi.h5')
    parser.add_argument('--h5_file', type=str, default=default_h5_file)
    now = str(datetime.now().strftime('%y-%m-%d %H:%M:%S'))
    default_model_path = os.path.join(project_path,
                                      'trained_models/model_{}/'.format(now))
    parser.add_argument('--output_dir', type=str, default=default_model_path)
    parser.add_argument('--random_seed', type=int, default=0)
    return parser.parse_args()


def _check_for_h5_file(h5_file):
    """Checks for a H5 file. If unavailable, downloads/creates files from SabDab."""
    if not os.path.isfile(h5_file):
        project_path = os.path.abspath(os.path.join(deepscab.__file__,
                                                    "../.."))
        ab_dir = os.path.join(project_path, 'data/antibody_database')
        print('No H5 file found at {}, creating new file in {}/ ...'.format(
            h5_file, ab_dir))
        if not os.path.isdir(ab_dir):
            print('{}/ does not exist, creating {}/ ...'.format(
                ab_dir, ab_dir))
            os.system(f"mkdir -p {ab_dir}")
        pdb_files = [f.endswith('pdb') for f in os.listdir(ab_dir)]
        if len(pdb_files) == 0:
            print('No PDB files found in {}, downloading PDBs ...'.format(
                ab_dir))
            download_train_dataset()
        print('Creating new h5 file at {} using data from {}/ ...'.format(
            h5_file, ab_dir))
        antibody_to_h5(ab_dir, h5_file, print_progress=True)


def _cli():
    """Command line interface for train.py when it is run as a script"""
    args = _get_args()
    properties = dict(num_out_bins=args.num_bins,
                      num_blocks1D=args.num_blocks1D,
                      num_blocks2D=args.num_blocks2D,
                      dropout_proportion=args.dropout,
                      dilation_cycle=args.dilation_cycle)
    model = AbChiResNet(21, **properties)
    device_type = 'cuda' if torch.cuda.is_available(
    ) and args.try_gpu else 'cpu'
    device = torch.device(device_type)

    optimizer = Adam(model.parameters(), lr=args.lr)
    properties.update({'lr': args.lr})
    criterion = nn.CrossEntropyLoss(ignore_index=-999)

    # Load dataset loaders from h5 file
    h5_file = args.h5_file
    _check_for_h5_file(h5_file)
    dataset = H5AbChiDataset(h5_file, num_bins=args.num_bins)
    train_split_length = int(len(dataset) * args.train_split)
    torch.manual_seed(args.random_seed)
    train_dataset, validation_dataset = data.random_split(
        dataset, [train_split_length,
                  len(dataset) - train_split_length])
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=H5AbChiDataset.merge_samples_to_minibatch)
    validation_loader = data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        collate_fn=H5AbChiDataset.merge_samples_to_minibatch)

    lr_modifier = ReduceLROnPlateau(optimizer, verbose=True)
    out_dir = args.output_dir
    if not os.path.isdir(out_dir):
        print('Making {} ...'.format(out_dir))
        os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(out_dir, 'tensorboard'))

    print('Arguments:\n', args)
    print('Model:\n', model)

    train(model=model,
          train_loader=train_loader,
          validation_loader=validation_loader,
          optimizer=optimizer,
          device=device,
          epochs=args.epochs,
          criterion=criterion,
          lr_modifier=lr_modifier,
          writer=writer,
          save_file=os.path.join(out_dir, 'model.p'),
          properties=properties)


if __name__ == '__main__':
    _cli()
