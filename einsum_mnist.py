from argparse import ArgumentParser ;
import os,sys, math
import numpy as np
import torch
from EinsumNetwork import Graph, EinsumNetwork
import datasets
import utils
import json, random ;

def computeLeafClass(id):
  exponential_family = None ;
  if id == "binomial":
    exponential_family = EinsumNetwork.BinomialArray
  elif id == "categorical":
    exponential_family = EinsumNetwork.CategoricalArray
  elif id == "normal":
    exponential_family = EinsumNetwork.NormalArray
  return exponential_family ;


def outliers_batched(einet, in_pt, out_pt, batch_size=100):
    """Computes log-likelihood in batched way."""
    inlier_list = [] ;
    outlier_list = [] ;
    with torch.no_grad():
        idx_batches_in = torch.arange(0, in_pt.shape[0], dtype=torch.int64, device=in_pt.device).split(batch_size)
        idx_batches_out = torch.arange(0, out_pt.shape[0], dtype=torch.int64, device=out_pt.device).split(batch_size)
        ll_total = 0.0
        for batch_count, idx in enumerate(idx_batches_in):
            batch_in = in_pt[idx, :]
            in_logprobs = einet(batch_in)
            lls_in = EinsumNetwork.log_likelihoods(in_logprobs, None)
            inlier_list.append(lls_in.cpu().numpy())

        for batch_count, idx in enumerate(idx_batches_out):
            batch_out = out_pt[idx, :]
            out_logprobs = einet(batch_out)
            lls_out = EinsumNetwork.log_likelihoods(out_logprobs, None)
            outlier_list.append(lls_out.cpu().numpy())

    return np.concatenate(inlier_list), np.concatenate(outlier_list)



device = 'cuda' if torch.cuda.is_available() else 'cpu'

demo_text = """
This demo loads (fashion) mnist and quickly trains an EiNet for some epochs. 

There are some parameters to play with, as for example which exponential family you want 
to use, which classes you want to pick, and structural parameters. Then an EiNet is trained, 
the log-likelihoods reported, some (conditional and unconditional) samples are produced, and
approximate MPE reconstructions are generated. 
"""
print(demo_text)

############################################################################
############################################################################
parser = ArgumentParser() ;
parser.add_argument("--exp_id",type=str, required=True) ;
parser.add_argument("--train",type=eval, default=False, required=False) ;
parser.add_argument("--fashion_mnist",type=eval, default=False, required=False) ;
parser.add_argument("--epochs",type=int, default=20, required=False) ;
parser.add_argument("--batch_size",type=int, default=100, required=False) ;
parser.add_argument("--depth",type=int, default=100, required=False) ;
parser.add_argument("--num_repetitions",type=int, default=30, required=False) ;
parser.add_argument("--pd_num_pieces",type=int, default=4, required=False) ;
parser.add_argument("--num_samples",type=int, default=100, required=False) ;
parser.add_argument("--online_em_frequency",type=int, default=1, required=False) ;
parser.add_argument("--online_em_stepsize",type=float, default=0.1, required=False) ;
parser.add_argument("--K",type=int, default=30, required=False) ;
parser.add_argument("--outlier_classes",type=int, nargs="*", default=[0], required=False) ;
parser.add_argument("--classes",type=int, nargs="*",default=[1,2,3,4,5,6,7,8,9], required=False) ;
parser.add_argument("--structure",type=str, default="poon-domingos", required=False) ;
parser.add_argument("--exponential_family",type=str, default="binomial", required=False, choices=["binomial", "normal","categorical"]) ;
FLAGS = parser.parse_args(sys.argv[1:]) ;
if type(FLAGS.classes) ==  type(1):
  FLAGS.classes = [FLAGS.classes] ;
if type(FLAGS.outlier_classes) ==  type(1):
  FLAGS.outlier_classes = [FLAGS.outlier_classes] ;

print (FLAGS.outlier_classes)

fashion_mnist = FLAGS.fashion_mnist ;

exponential_family = computeLeafClass(FLAGS.exponential_family) ;

classes = FLAGS.classes ;

K = FLAGS.K ;

structure = FLAGS.structure ;
#structure = 'binary-trees'

# 'poon-domingos'
pd_num_pieces = FLAGS.pd_num_pieces ;
# pd_num_pieces = [7]
# pd_num_pieces = [7, 28]
width = 28
height = 28

# 'binary-trees'
depth = FLAGS.depth ;
num_repetitions = FLAGS.num_repetitions ;

num_epochs = FLAGS.epochs ;
batch_size = FLAGS.batch_size ;
online_em_frequency = FLAGS.online_em_frequency ;
online_em_stepsize = FLAGS.online_em_stepsize ;

###########################################

exponential_family_args = None
if exponential_family == EinsumNetwork.BinomialArray:
    exponential_family_args = {'N': 255}
if exponential_family == EinsumNetwork.CategoricalArray:
    exponential_family_args = {'K': 256}
if exponential_family == EinsumNetwork.NormalArray:
    exponential_family_args = {'min_var': 1e-6, 'max_var': 0.1}

# get data
if FLAGS.fashion_mnist:
    train_x, train_labels, test_x, test_labels = datasets.load_fashion_mnist()
else:
    train_x, train_labels, test_x, test_labels = datasets.load_mnist()

if exponential_family == EinsumNetwork.NormalArray:
    train_x /= 255.
    test_x /= 255.
    #train_x -= .5
    #test_x -= .5
else:
    pass ;
    train_x /= 255.
    test_x /= 255.
    

# validation split
valid_x = train_x[-10000:, :]
train_x = train_x[:-10000, :]
valid_labels = train_labels[-10000:]
train_labels = train_labels[:-10000]

# pick the selected outlier classes
if FLAGS.outlier_classes != []:
    out_train_x = train_x[np.any(np.stack([train_labels == c for c in FLAGS.outlier_classes], 1), 1), :]
    out_valid_x = valid_x[np.any(np.stack([valid_labels == c for c in FLAGS.outlier_classes], 1), 1), :]
    out_test_x = test_x[np.any(np.stack([test_labels == c for c in FLAGS.outlier_classes], 1), 1), :]
    out_train_x = torch.from_numpy(out_train_x).to(torch.device(device))
    out_valid_x = torch.from_numpy(out_valid_x).to(torch.device(device))
    out_test_x = torch.from_numpy(out_test_x).to(torch.device(device))


# pick the selected classes
if FLAGS.classes is not None:
    train_x = train_x[np.any(np.stack([train_labels == c for c in FLAGS.classes], 1), 1), :]
    valid_x = valid_x[np.any(np.stack([valid_labels == c for c in FLAGS.classes], 1), 1), :]
    test_x = test_x[np.any(np.stack([test_labels == c for c in FLAGS.classes], 1), 1), :]


train_x = torch.from_numpy(train_x).to(torch.device(device))
valid_x = torch.from_numpy(valid_x).to(torch.device(device))
test_x = torch.from_numpy(test_x).to(torch.device(device))


# Make EinsumNetwork
######################################
if FLAGS.structure == 'poon-domingos':
    # Use only a single hierarchy level as in the einsum paper for SVHN
    pd_delta = [height,width / pd_num_pieces]
    graph = Graph.poon_domingos_structure(shape=(height, width), axes=[1], delta=[width/FLAGS.pd_num_pieces])
elif FLAGS.structure == 'binary-trees':
    graph = Graph.random_binary_trees(num_var=train_x.shape[1], depth=FLAGS.depth, num_repetitions=FLAGS.num_repetitions)
else:
    raise AssertionError("Unknown Structure")

args = EinsumNetwork.Args(
        num_var=train_x.shape[1],
        num_dims=1,
        num_classes=1,
        num_sums=FLAGS.K,
        num_input_distributions=FLAGS.K,
        exponential_family=exponential_family,
        exponential_family_args=exponential_family_args,
        online_em_frequency=online_em_frequency,
        online_em_stepsize=online_em_stepsize)

einet = EinsumNetwork.EinsumNetwork(graph, args)
einet.initialize()
einet.to(device)
print(einet)
#summary(einet,input_size=(1,28,28) )
acc = 0 ;
for p in einet.parameters():
    # p.requires_grad: bool
    # p.data: Tensor
    print(p.data.shape)
    acc += np.prod(p.data.shape)
print ("# parameters is ", acc) 

train_N = train_x.shape[0]
valid_N = valid_x.shape[0]
test_N = test_x.shape[0]

# prepare paths
if FLAGS.fashion_mnist:
  model_dir = './models_einsum'
  samples_dir = './results'
else:
  model_dir = './models_einsum'
  samples_dir = './results'
utils.mkdir_p(model_dir)
utils.mkdir_p(samples_dir)
model_file = os.path.join(model_dir, FLAGS.exp_id+"_einet.mdl")



if FLAGS.train == True:
  # Train
  ######################################


  for epoch_count in range(FLAGS.epochs):

    ##### evaluate
    einet.eval()
    train_ll = EinsumNetwork.eval_loglikelihood_batched(einet, train_x, batch_size=batch_size)
    valid_ll = EinsumNetwork.eval_loglikelihood_batched(einet, valid_x, batch_size=batch_size)
    test_ll = EinsumNetwork.eval_loglikelihood_batched(einet, test_x, batch_size=batch_size)
    print("[{}]   train LL {}   valid LL {}   test LL {}".format(
        epoch_count,
        train_ll / train_N,
        valid_ll / valid_N,
        test_ll / test_N))
    einet.train()
    #####

    idx_batches = torch.randperm(train_N, device=device).split(batch_size)

    total_ll = 0.0
    for idx in idx_batches:
        batch_x = train_x[idx, :]
        outputs = einet.forward(batch_x)
        ll_sample = EinsumNetwork.log_likelihoods(outputs)
        log_likelihood = ll_sample.sum()
        log_likelihood.backward()

        einet.em_process_batch()
        total_ll += log_likelihood.detach().item()

    einet.em_update()

  # save model if it was trained peviously
  graph_file = os.path.join(model_dir, FLAGS.exp_id+"_einet.pc")
  Graph.write_gpickle(graph, graph_file)
  print("Saved PC graph to {}".format(graph_file))
  torch.save(einet, model_file)
  print("Saved model to {}".format(model_file))


# reload model
if FLAGS.train == False:
  del einet
  einet = torch.load(model_file)
  print("Loaded model from {}".format(model_file))


#####################
# draw some samples #
#####################

# Draw conditional samples for reconstruction
image_scope = np.array(range(height * width)).reshape(height, width)
marginalize_idx = list(image_scope[0:round(height/2), :].reshape(-1))
keep_idx = [i for i in range(width*height) if i not in marginalize_idx]
einet.set_marginalization_idx(marginalize_idx)

num_samples = 10
samples = None
for k in range(num_samples):
    if samples is None:
        samples = einet.sample(x=test_x[0:25, :]).cpu().numpy()
    else:
        samples += einet.sample(x=test_x[0:25, :]).cpu().numpy()
samples /= num_samples
samples = samples.squeeze()

samples = samples.reshape((-1, 28, 28))
utils.save_image_stack(samples, 5, 5, os.path.join(samples_dir, FLAGS.exp_id+"_sample_reconstruction.png"), margin_gray_val=0.)

# ground truth
ground_truth = test_x[0:25, :].cpu().numpy()
ground_truth = ground_truth.reshape((-1, 28, 28))
utils.save_image_stack(ground_truth, 5, 5, os.path.join(samples_dir, FLAGS.exp_id+"_ground_truth.png"), margin_gray_val=0.)

###############################
# perform mpe reconstructions #
###############################

mpe = einet.mpe().cpu().numpy()
mpe = mpe.reshape((1, 28, 28))
utils.save_image_stack(mpe, 1, 1, os.path.join(samples_dir, FLAGS.exp_id+"_mpe.png"), margin_gray_val=0.)

# Draw conditional samples for reconstruction
image_scope = np.array(range(height * width)).reshape(height, width)
marginalize_idx = list(image_scope[0:round(height/2), :].reshape(-1))
keep_idx = [i for i in range(width*height) if i not in marginalize_idx]
einet.set_marginalization_idx(marginalize_idx)

mpe_reconstruction = einet.mpe(x=test_x[0:25, :]).cpu().numpy()
mpe_reconstruction = mpe_reconstruction.squeeze()
mpe_reconstruction = mpe_reconstruction.reshape((-1, 28, 28))
utils.save_image_stack(mpe_reconstruction, 5, 5, os.path.join(samples_dir, FLAGS.exp_id+"_mpe_reconstruction.png"), margin_gray_val=0.)

print()
print('Saved samples to {}'.format(samples_dir))

####################
# save and re-load #
####################

# evaluate log-likelihoods
einet.eval()
train_ll_before = EinsumNetwork.eval_loglikelihood_batched(einet, train_x, batch_size=batch_size)
valid_ll_before = EinsumNetwork.eval_loglikelihood_batched(einet, valid_x, batch_size=batch_size)
test_ll_before = EinsumNetwork.eval_loglikelihood_batched(einet, test_x, batch_size=batch_size)

# unconditional sampling along with fake classes 
if FLAGS.num_samples > 0:
  samples = einet.sample(num_samples=FLAGS.num_samples).cpu().numpy()
  samples = samples.reshape((FLAGS.num_samples, 28,28))
  #samples /=
  #samples += 0.5 ;
  d = int(math.sqrt(FLAGS.num_samples)) ; # assume num_sampels is a square number
  print ("Sampples ahve shape ", samples.shape) ;
  utils.save_image_stack(samples, d, d, os.path.join(samples_dir, FLAGS.exp_id+"_samples.png"), margin_gray_val=0.)
  samples = samples.reshape((FLAGS.num_samples, 28,28,1))
  np.save("./results/"+FLAGS.exp_id+"_samples.npy", samples) ;
  # generate fake class labels from "classes" list
  nr_classes = len(FLAGS.classes) ;
  indices = [random.randint(0,nr_classes-1) for i in range(0,FLAGS.num_samples)] ; print("i=", indices) ;
  labels = np.array([FLAGS.classes[i] for i in indices]) ;
  oh_labels = np.zeros([FLAGS.num_samples, 10], dtype=np.float32) ;
  oh_labels [range(0,FLAGS.num_samples), labels] = 1.;
  np.save("./results/"+FLAGS.exp_id+"_labels.npy", oh_labels) ;

# outlier detection
if FLAGS.outlier_classes != []:
  #ll_test_in  = EinsumNetwork.log_likelihoods(einet(test_x)).cpu().detach().numpy()
  #ll_test_out  = EinsumNetwork.log_likelihoods(einet(out_test_x)).cpu().detach().numpy() 

  ll_test_in, ll_test_out = outliers_batched(einet, test_x, out_test_x)
  print (ll_test_in.shape)

  # store raw (not neg) logliks to json file in a specific format
  # apparently we use neg loglik for training but loglik otherwise
  js = {"eval":{}}
  js["eval"].update({"scores_T1_outliers":[0,[0,1,[float(ll[0]) for ll in ll_test_in]]]})
  js["eval"].update({"scores_T2_outliers":[0,[0,1,[float(ll[0]) for ll in ll_test_out]]]})
  f = open(os.path.join(samples_dir, FLAGS.exp_id+"_log.json"),"w")
  json.dump(js, f) ;
  f.close() ;




# evaluate log-likelihoods on re-loaded model
train_ll = EinsumNetwork.eval_loglikelihood_batched(einet, train_x, batch_size=batch_size)
valid_ll = EinsumNetwork.eval_loglikelihood_batched(einet, valid_x, batch_size=batch_size)
test_ll = EinsumNetwork.eval_loglikelihood_batched(einet, test_x, batch_size=batch_size)
print()
print("Log-likelihoods before saving --- train LL {}   valid LL {}   test LL {}".format(
        train_ll / train_N,
        valid_ll / valid_N,
        test_ll / test_N))
print("Log-likelihoods after saving  --- train LL {}   valid LL {}   test LL {}".format(
        train_ll / train_N,
        valid_ll / valid_N,
        test_ll / test_N))
