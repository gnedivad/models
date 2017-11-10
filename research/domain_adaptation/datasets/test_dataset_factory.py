import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from domain_adaptation.datasets import dataset_factory


def main():
  source_dataset = dataset_factory.get_dataset(
      'lung_m',
      split_name='float32',
      dataset_dir='C:/tmp/data')
  source_real_images, source_virtual_images, source_labels = (
    dataset_factory.provide_batch(
      'lung_m', 'float32', 'C:/tmp/data', 4, 32, 4
    )
  )

  once = False
  with tf.Session() as sess:
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
    try:
      while not coordinator.should_stop() and not once:
        results = sess.run(fetches={
          'source_real_images': source_real_images,
          'source_virtual_images': source_virtual_images,
          'source_labels': source_labels,
        })
        once = True
    except Exception as e:
      print(e); return
    finally:
      coordinator.request_stop()

  # print(results['source_real_images'].shape)
  # plt.imshow(np.squeeze(results['source_real_images'][0]), cmap='gray')

  fig, ax = plt.subplots(nrows=4, ncols=8)
  for i in range(12):  # len(real_images)):
    ax[i%4,i//4*3].imshow(np.squeeze(results['source_real_images'][i]), cmap='gray')
    ax[i%4,i//4*3+1].imshow(np.squeeze(results['source_virtual_images'][i]), cmap='gray')
    ax[i%4,i//4*3].axis("off")
    ax[i%4,i//4*3+1].axis("off")
    if i//4*3+2 < 8:
      ax[i%4,i//4*3+2].axis("off")

  plt.show()
  print(results['source_labels'][:12])

if __name__ == "__main__":
  main()