import argparse
import os

import numpy as np
import torch as t
from torch.optim import Adam

from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from model.rvae import RVAE

if __name__ == "__main__":

    if not os.path.exists('data/word_embeddings.npy'):
        raise FileNotFoundError("word embeddings file was't found")

    parser = argparse.ArgumentParser(description='RVAE')
    parser.add_argument('--num-iterations', type=int, default=100000, metavar='NI',
                        help='num iterations (default: 120000)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                        help='batch size (default: 32)')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--learning-rate', type=float, default=0.00005, metavar='LR',
                        help='learning rate (default: 0.00005)')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='DR',
                        help='dropout (default: 0.3)')
    parser.add_argument('--use-trained', type=bool, default=False, metavar='UT',
                        help='load pretrained model (default: False)')
    parser.add_argument('--ce-result', default='', metavar='CE',
                        help='ce result path (default: '')')
    parser.add_argument('--kld-result', default='', metavar='KLD',
                        help='ce result path (default: '')')

    args = parser.parse_args()

    batch_loader = BatchLoader('')
    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.words_vocab_size,
                            batch_loader.chars_vocab_size)

    rvae = RVAE(parameters)
    if args.use_trained:
        rvae.load_state_dict(t.load('trained_RVAE'))
    if args.use_cuda:
        rvae = rvae.cuda()

    optimizer = Adam(rvae.learnable_parameters(), args.learning_rate)

    train_step = rvae.trainer(optimizer, batch_loader)
    validate = rvae.validater(batch_loader)

    train_ce_result = []
    train_kld_result = []
    train_kld_coef = []

    ce_result = []
    kld_result = []

    for iteration in range(args.num_iterations):

        try:
            cross_entropy_t, kld_t, coef = train_step(iteration, args.batch_size, args.use_cuda, args.dropout)

            if iteration % 5 == 0:
                print('\n')
                print('------------TRAIN-------------')
                print('----------ITERATION-----------')
                print(iteration)
                print('--------CROSS-ENTROPY---------')
                print(cross_entropy_t.data.cpu().numpy())
                print('-------------KLD--------------')
                print(kld_t.data.cpu().numpy())
                print('-----------KLD-coef-----------')
                print(coef)
                print('------------------------------')

            if iteration % 10 == 0:
                cross_entropy, kld = validate(args.batch_size, args.use_cuda)

                cross_entropy = cross_entropy.data.cpu().numpy()
                kld = kld.data.cpu().numpy()
                cross_entropy_t = cross_entropy_t.data.cpu().numpy()
                kld_t = kld_t.data.cpu().numpy()

                print('\n')
                print('------------VALID-------------')
                print('--------CROSS-ENTROPY---------')
                print(cross_entropy)
                print('-------------KLD--------------')
                print(kld)
                print('------------------------------')

                ce_result += [cross_entropy]
                kld_result += [kld]

                train_ce_result += [cross_entropy_t]
                train_kld_result += [kld_t]
                train_kld_coef += [coef]

                del cross_entropy, kld

            if iteration % 20 == 0:
                seed = np.random.normal(size=[1, parameters.latent_variable_size])

                sample = rvae.sample(batch_loader, 50, seed, args.use_cuda)

                print('\n')
                print('------------SAMPLE------------')
                print('------------------------------')
                print(sample)
                print('------------------------------')
            del cross_entropy_t, kld_t, coef
        except KeyboardInterrupt:
            print('Training stopped by user!')
            break

    from matplotlib import pyplot as plt
    plt.figure()
    plt.title('Cross-Entropy Losses')
    plt.plot(ce_result, 'r-', label='valid-cross-entropy')
    plt.plot(train_ce_result, 'b-', label="train-cross-entropy")
    plt.legend()
    plt.savefig('data/CE-losses.png')
    plt.show()

    plt.figure()
    plt.title('KL-Divergence Losses')
    plt.plot(kld_result, 'r-', label='valid-KL-Divergence')
    plt.plot(train_kld_result, 'b-', label='train-KL-Divergence')
    plt.legend()
    plt.savefig('data/KL-losses.png')
    plt.show()

    plt.figure()
    plt.title('KL-coefficient')
    plt.plot(train_kld_coef, 'r-')
    plt.savefig('data/KL-coefficient')
    plt.show()

    # t.save(rvae.state_dict(), 'trained_RVAE')
    t.save(rvae.state_dict(), 'trained_RVAE_code')

    np.save('ce_result_{}.npy'.format(args.ce_result), np.array(ce_result))
    np.save('kld_result_npy_{}'.format(args.kld_result), np.array(kld_result))
