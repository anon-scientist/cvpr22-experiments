# convolutional and non-convolutional genSPN for sampling and outlier detection
# Adapted from  https://github.com/pronobis/libspn-keras/blob/master/examples/notebooks/Sampling%20with%20conv%20SPNs.ipynb
# Adapted by AG for unsupervised sampling and outlier det., argparse support, loading/saving etc.
import libspn_keras as spnk
from tensorflow import keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from argparse import ArgumentParser ;
import sys, json, math, random ;



import numpy as np
import tensorflow_datasets as tfds
from libspn_keras.layers import NormalizeAxes
import tensorflow as tf


def to_float32(x):
  return tf.cast(tf.reshape(x,(28,28,1)),tf.float32)


def build_spn_conv_mn(sum_op, return_logits=False, infer_no_evidence=False, unsupervised=True, location_initializer=None, base_K = 2):
    spnk.set_default_sum_op(sum_op)
    spnk.set_default_linear_accumulators_constraint(spnk.constraints.GreaterEqualEpsilon())
    stack = [
        spnk.layers.NormalizeStandardScore(input_shape=(28,28, 1), normalization_epsilon = 0.1),
        # Non-overlapping products
        spnk.layers.NormalLeaf(
            num_components=4, 
            location_trainable=True,
            location_initializer=location_initializer,
            use_accumulators=True,
            scale_trainable=False
        ),
        tf.keras.layers.Dropout(rate=0.1),
        spnk.layers.Conv2DProduct(name="p1",
            depthwise=False, 
            strides=[1, 1], 
            dilations=[1, 1], 
            kernel_size=[2, 2],
            padding='full', 
            #num_channels = 24
        ),
        spnk.layers.Local2DSum(num_sums=base_K),
        # Non-overlapping products
        spnk.layers.Conv2DProduct(name="p2", 
            depthwise=True, 
            strides=[1, 1], 
            dilations=[2, 2], 
            kernel_size=[2, 2],
            padding='full', 
            num_channels = None
        ),
        spnk.layers.Local2DSum(num_sums=base_K),
        # Non-overlapping products
        spnk.layers.Conv2DProduct(name="p3",
            depthwise=True, 
            strides=[1, 1], 
            dilations=[4, 4], 
            kernel_size=[2, 2],
            padding='full', 
            num_channels = None
        ),
        spnk.layers.Local2DSum(num_sums=base_K),
        spnk.layers.Conv2DProduct(name="p4",
            depthwise=True, 
            strides=[1, 1], 
            dilations=[8, 8], 
            kernel_size=[2, 2],
            padding='full',
            num_channels = 1024
        ),
        spnk.layers.Local2DSum(num_sums=base_K),
        spnk.layers.Conv2DProduct(name="p5",
            depthwise=True, 
            strides=[1, 1], 
            dilations=[16, 16], 
            kernel_size=[2, 2],
            padding='full',
            num_channels = 1024
        ),
        spnk.layers.Local2DSum(num_sums=base_K),
        spnk.layers.Conv2DProduct(name="p6",
            depthwise=True, 
            strides=[1, 1], 
            dilations=[32, 32], 
            kernel_size=[2, 2],
            padding='full',
            num_channels = 1024
        ),
        spnk.layers.Local2DSum(num_sums=base_K),
        spnk.layers.Conv2DProduct(name="p7",
            depthwise=True, 
            strides=[1, 1], 
            dilations=[64, 64], 
            kernel_size=[2, 2],
            padding='final',
            num_channels = 1024
        ),
        spnk.layers.LogDropout(rate=0.1),
        spnk.layers.SpatialToRegions(),
        #spnk.layers.DenseProduct(num_factors = 1),
        spnk.layers.RootSum(
            return_weighted_child_logits=return_logits
        )
    ]
    sum_product_network = spnk.models.SequentialSumProductNetwork(
      stack, infer_no_evidence=infer_no_evidence, unsupervised=unsupervised)
    return sum_product_network ;


def build_spn_nonconv(sum_op, return_logits=False, infer_no_evidence=False, unsupervised=True, location_initializer=None, normalize_layer=None, base_K=256):
  if normalize_layer is None:
    normnalize_layer = spnk.layers.NormalizeStandardScore(input_shape=(28,28, 1))  ;
  spnk.set_default_sum_op(sum_op)
  return spnk.models.SequentialSumProductNetwork([
    normalize_layer,
    spnk.layers.NormalLeaf(
        num_components=8, 
        location_trainable=True,
        location_initializer=location_initializer,
        scale_trainable=True
    ),
    spnk.layers.Conv2DProduct(
        depthwise=False, 
        strides=[2, 2], 
        dilations=[1, 1], 
        kernel_size=[2, 2],
        padding='valid'
    ),
    spnk.layers.Local2DSum(num_sums=base_K),
    spnk.layers.Conv2DProduct(
        depthwise=True, 
        strides=[2, 2], 
        dilations=[1, 1], 
        kernel_size=[2, 2],
        padding='valid'
    ),
    spnk.layers.Local2DSum(num_sums=base_K * 2),
    # Pad to go from 7x7 to 8x8, so that we can apply 3 more Conv2DProducts
    tf.keras.layers.ZeroPadding2D(((0, 1), (0, 1))),
    spnk.layers.Conv2DProduct(
        depthwise=True, 
        strides=[2, 2], 
        dilations=[1, 1], 
        kernel_size=[2, 2],
        padding='valid'
    ),
    spnk.layers.Local2DSum(num_sums=base_K * 2),
    spnk.layers.Conv2DProduct(
        depthwise=True, 
        strides=[2, 2], 
        dilations=[1, 1], 
        kernel_size=[2, 2],
        padding='valid'
    ),
    spnk.layers.Local2DSum(num_sums=base_K * 4),
    spnk.layers.Conv2DProduct(
        depthwise=True, 
        strides=[2, 2], 
        dilations=[1, 1], 
        kernel_size=[2, 2],
        padding='valid'
    ),
    spnk.layers.LogDropout(rate=0.5),
    #spnk.layers.DenseSum(num_sums=10) #,
    spnk.layers.RootSum(return_weighted_child_logits=return_logits)
  ], infer_no_evidence=infer_no_evidence, unsupervised=unsupervised)

def selectClasses(x,y,classes):
  if classes == []:
    return None, None ;

  if type(classes) == type(1):
    classes = [classes] ;

  #print (tf.reduce_max(x), "mayx") ;
  masks = [tf.cast((y == c), tf.int32) for c in classes]
  mask = masks[0] ;
  for m in masks[1:]:
    mask = mask + m ;

  indices = tf.reshape(tf.where(mask>=1),(-1)) ;

  select_x = tf.reshape(tf.cast(tf.gather(x, indices),tf.float32),(-1,28,28,1)) / 255. ;
  select_y = tf.gather(y, indices) ;

  return select_x, select_y ;

if __name__ == "__main__":

  parser = ArgumentParser() ;
  parser.add_argument("--train",type=eval, default=False, required=False) ;
  parser.add_argument("--plot_samples",type=eval, default=False, required=False) ;
  parser.add_argument("--exp_id",type=str, required=True) ;
  parser.add_argument("--fashion_mnist",type=eval, default=False, required=False) ;
  parser.add_argument("--epochs",type=int, default=20, required=False) ;
  parser.add_argument("--num_samples",type=int, default=100, required=False) ;
  parser.add_argument("--base_K",type=int, default=16, required=False, help = 'num_sums base value') ;
  parser.add_argument("--learning_rate",type=float, default=0.05, required=False) ;
  parser.add_argument("--batch_size",type=int, default=100, required=False) ;
  parser.add_argument("--K",type=int, default=30, required=False) ;
  parser.add_argument("--outlier_classes",type=int, nargs="*", default=[], required=False) ;
  parser.add_argument("--classes",type=int, nargs="*",default=[1,2,3,4,5,6,7,8,9], required=False) ;
  parser.add_argument("--structure",type=str, default="nonconv", choices=["conv","nonconv"],required=False) ;
  parser.add_argument("--checkpoint",type=str, default="test",required=False) ;
  FLAGS = parser.parse_args(sys.argv[1:]) ;

  spnk.set_default_accumulator_initializer(
      spnk.initializers.Dirichlet(alpha=0.1)
  )

  normalize = spnk.layers.NormalizeStandardScore(
    input_shape=(28, 28, 1), axes=NormalizeAxes.GLOBAL, 
    normalization_epsilon=1e-3
  )

  # load dataset, extract train and outlier/test classes, convert to float2 etc...
  #  works for mnist, fmnist, ?
  dataset_name = "mnist" if FLAGS.fashion_mnist == False else "fashion_mnist" ;
  
  x_train, y_train = tfds.load(name=dataset_name, batch_size=-1, split="train", as_supervised=True)
  x_test, y_test = tfds.load(name=dataset_name, batch_size=-1, split="test", as_supervised=True)
  select_x_train,select_y_train = selectClasses(x_train,y_train,FLAGS.classes) ;
  select_x_test,select_y_test = selectClasses(x_test,y_test,FLAGS.classes) ;
  select_x_test_outliers,select_y_test_outliers = selectClasses(x_test,y_test,FLAGS.outlier_classes) ;
  print(select_x_train.shape, "XXXX") 

  # first create an MNIST dataset where all pixels are normalized to their dataset-wide mean and stddev
  preview = tf.data.Dataset.from_tensor_slices(select_x_train).batch(batch_size=32,drop_remainder=True) ;
  normalize.adapt(preview) 
  mnist_normalized = preview.map(normalize)

  # initialize SPN, using the statistics computed before
  location_initializer = spnk.initializers.PoonDomingosMeanOfQuantileSplit(
      mnist_normalized
  )

  # construct network as unsupervised, apparently infzer_no_evidence must be false in this case. Beats me why ...
  if FLAGS.structure == "nonconv":
    sum_product_network = build_spn_nonconv(spnk.SumOpEMBackprop(), return_logits=False, infer_no_evidence=False,location_initializer=location_initializer, normalize_layer=normalize, base_K = FLAGS.base_K)
  else:
    sum_product_network = build_spn_conv_mn(spnk.SumOpEMBackprop(), return_logits=False, infer_no_evidence=False, location_initializer=location_initializer, base_K = FLAGS.base_K)
  sum_product_network.summary()

  # prepare mnist in mini-batches, split train/test
  batch_size = FLAGS.batch_size

  mnist_train = tf.data.Dataset.from_tensor_slices((select_x_train,select_y_train)).shuffle(1024).batch(batch_size)

  mnist_test = tf.data.Dataset.from_tensor_slices((select_x_test, select_y_test)).batch(100)


  # construct SPN for trainign.  Select loglik as loss since all is unsupervised
  optimizer = spnk.optimizers.OnlineExpectationMaximization(learning_rate=FLAGS.learning_rate)
  metrics = [spnk.metrics.LogLikelihood()]
  loss = spnk.losses.NegativeLogLikelihood()
  sum_product_network.compile(loss=loss, metrics=metrics, optimizer=optimizer)


  # training!
  if FLAGS.train == True:
    epochs = FLAGS.epochs ;
    sum_product_network.fit(mnist_train, epochs=epochs, callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", min_delta=0.1, patience=2, factor=0.5)])
    sum_product_network.evaluate(mnist_test)
    sum_product_network.save_weights(FLAGS.checkpoint)
  print ("train done") ;

  # construct sampling SPN
  if FLAGS.structure == "nonconv":
    sum_product_network_sample = build_spn_nonconv(spnk.SumOpSampleBackprop(), return_logits=False, infer_no_evidence=True, unsupervised=False, location_initializer=location_initializer, normalize_layer=normalize, base_K = FLAGS.base_K)
  else:
    sum_product_network_sample = build_spn_conv_mn(spnk.SumOpSampleBackprop(), return_logits=False, infer_no_evidence=True, unsupervised=False,location_initializer=location_initializer, base_K = FLAGS.base_K)

  print ("load") ;
  sum_product_network_sample.load_weights(FLAGS.checkpoint)
  sum_product_network.load_weights(FLAGS.checkpoint)
  print ("load done") ;


  # outliers!
  if FLAGS.outlier_classes != []:
    print("Computing outliers") ;
    ll_test_in = sum_product_network.predict(select_x_test) ;
    ll_test_out = sum_product_network.predict(select_x_test_outliers) ;
    js = {"eval":{}}
    js["eval"].update({"scores_T1_outliers":[0,[0,1,[float(ll[0]) for ll in ll_test_in]]]})
    js["eval"].update({"scores_T2_outliers":[0,[0,1,[float(ll[0]) for ll in ll_test_out]]]})
    f = open("./results/"+FLAGS.exp_id+"_log.json","w")
    json.dump(js,f) ;
    f.close() ; 
    print("Computing outliers done") ;
  


  # sample!
  print("Sampling images: ", FLAGS.num_samples)
  samples = sum_product_network_sample.zero_evidence_inference(FLAGS.num_samples)
  print("samples minmax=",samples.min(), samples.max()) ;
  n = int(math.sqrt(FLAGS.num_samples))
  mn = samples.min() ; mx = samples.max() ;
  samples -= mn ; samples /= (mx-mn) ;

  # plot sampling results
  print("Sampling done... Now ploting results", samples.shape)
  if FLAGS.plot_samples:
    fig = plt.figure(figsize=(12., 12.))
    grid = ImageGrid(
      fig, 111,
      nrows_ncols=(n, n),
      axes_pad=0.1,
    )

    for ax, im in zip(grid, samples):
      ax.imshow(np.squeeze(im), cmap="gray")
    plt.savefig("./results/"+FLAGS.exp_id+"_samples.png")
  np.save( "./results/"+FLAGS.exp_id+"_samples.npy", samples.reshape(samples.shape[0],28,28,1),0,1) ;
  # generate fake class labels from "classes" list
  nr_classes = len(FLAGS.classes) ;
  indices = [random.randint(0,nr_classes-1) for i in range(0,FLAGS.num_samples)] ; print("i=", indices) ;
  labels = np.array([FLAGS.classes[i] for i in indices]) ;
  oh_labels = np.zeros([FLAGS.num_samples, 10], dtype=np.float32) ;
  oh_labels [range(0,FLAGS.num_samples), labels] = 1.;
  np.save("./results/"+FLAGS.exp_id+"_labels.npy", oh_labels) ;


  #plt.show()
