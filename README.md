# dlib - the lib for deep learning research

`dlib` is a collection of tools, utility functions, and modules that I have found useful for deep learning research, mostly biased towards training LLMs.

## Installation

The easiest way to use `dlib` is to simply copy all files into a root-level `dlib/` folder in your project. Alternatively, you can use `git subtree` to clone all files inside a root-level `dlib` folder and be able to easily pull new updates.

```bash
git subtree add --prefix ./dlib/ https://github.com/konstantinjdobler/dlib.git main --squash
```

The requirements are listed in `requiremnts.txt` but most of these are already present when doing NLP with `torch`. I have listed the "unusual" ones in `unusual-requirements.txt`.

## `git subtree`

Here are some commands that are useful when using `git subtree`.

Pulling new commits:

```bash
git subtree pull --prefix ./dlib/ https://github.com/konstantinjdobler/dlib.git main --squash
```

Pushing new commits. It's best to not mix changes in a subtree and the host repo in a single commit.

```bash
git subtree push --prefix ./dlib/ https://github.com/konstantinjdobler/dlib.git main
```

Useful resource: https://gist.github.com/SKempin/b7857a6ff6bddb05717cc17a44091202
