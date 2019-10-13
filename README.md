# Introduction
The code in this repo is a part of the blog post in Towards Data Science entitled [Throwing dice with maximum entropy principle](https://towardsdatascience.com/throwing-dice-with-maximum-entropy-principle-fa7707e72222).
## Generate images with docker container
```sh
sudo docker image build . -t maxent
sudo docker run -v $(pwd)/pics:/APP/pics maxent
```
## Tests:

To run tests:
```sh
$ cd tests
$ py.test
```

To regenerate tests:
```sh
$ cd tests
$ py.test --regtest-reset
```