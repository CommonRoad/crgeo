.. _contributing:

Contributing
======================

.. Everything in here is opinionated and subject to change. Suggestions welcome!

.. Code formatting
.. --------------

.. Testing priority
.. ---------------

.. As the core components and functionalities of commonroad-geometric have become quite stable and we aim to open source the framework, we will prioritize testing in the following manner:

.. 1.  Core framework (e.g. TrafficExtractor): Unit tests mandatory, reviewed very strictly

.. 2.  Extended framework (e.g. GeometricLearner): Unit tests recommended, reviewed

.. 3.  Implementations (i.e. everything in "implementations" folders): Unit tests optional, recommended for heavily used and complex components, usually not reviewed

.. 4.  Projects (i.e. everything in the "projects" folder): Almost anything goes

Commit Guideline
----------------

Don't forget to set your name and email
---------------------------------------

How can I change the author name / email of a commit? `https://www.git-tower.com/learn/git/faq/change-author-name-email <https://www.git-tower.com/learn/git/faq/change-author-name-email>`_

::

    $ git config user.name "John Doe"
    $ git config user.email "john@doe.org"

Commit Messages
---------------

A good commit message tells others what you did at a glance.

Without a good commit message, others have to look at all changes of your commit to judge whether some of them are relevant to their work.

We roughly follow the "The seven rules of a great Git commit message" of this blog post: `How to Write a Git Commit Message <https://cbea.ms/git-commit/>`_.

More specific, in descending order of importance:

1.  Use the imperative mood in the subject line
2.  Use the body to explain what and why vs. how (how is better explained in source code/comments)
3.  Capitalize the subject line
4.  Do not end the subject line with a period
5.  Separate subject from body with a blank line (if there's need for more than one line)

For examples, check the blog post or the history of the repository.

.. Merge requests
.. --------------

.. CI/CD pipeline
.. --------------

.. Determinism
.. -----------

.. *   "Reproducibility of results is a cornerstone of science"
.. *   Most of the experiments run using commonroad-geometric rely on random numbers and probabilistic methods, e.g. random exploration
.. *   A long term goal of this project is ensuring that 2 runs of the same experiment with the same `random seed <https://en.wikipedia.org/wiki/Random_seed>`_ produce the **exact same** results
.. *   Hence you should familiarize yourself with and observe our guideline for ensuring determinisme requests
.. --------------

.. CI/CD pipeline
.. --------------

.. Determinism
.. -----------

.. *   "Reproducibility of results is a cornerstone of science"
.. *   Most of the experiments run using commonroad-geometric rely on random numbers and probabilistic methods, e.g. random exploration
.. *   A long term goal of this project is ensuring that 2 runs of the same experiment with the same `random seed <https://en.wikipedia.org/wiki/Random_seed>`_ produce the **exact same** results
.. *   Hence e requests
.. --------------

.. CI/CD pipeline
.. --------------

.. Determinism
.. -----------

.. *   "Reproducibility of results is a cornerstone of science"
.. *   Most of the experiments run using commonroad-geometric rely on random numbers and probabilistic methods, e.g. random exploration
.. *   A long term goal of this project is ensuring that 2 runs of the same experiment with the same `random seed <https://en.wikipedia.org/wiki/Random_seed>`_ produce the **exact same** results
.. *   Hence 