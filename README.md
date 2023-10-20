# snnTorch learning

Personal playground to learn how `snnTorch` works and to create some interesting
models for the future.

It might not have the best practices, but it might help someone.

# Current status

* Iris dataset with SNN and 100% accuracy, original measures
* Iris dataset, but with `spikegen` used to create spikes for the measures
  * With rate coding, accuracy dropped to 46.67% due to lost information
  * When we do rate coding with linear interpolation, we're at 100% again (in this case, no need for `MinMaxScaler`)
  * Latency coding gives 46.67% again
  * Interestingly, delta coding gives 100% (no need for `MinMaxScaler`)
  * Note that the 100% numbers can vary from time to time by one sample, while this wasn't observer with original measures