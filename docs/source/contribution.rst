==================
Contribution Guide
==================

Thank you for considering contributing to our project! Your contributions are greatly appreciated. To ensure a smooth and effective contribution process, please follow the guidelines outlined below.

Setup Process
-------------

1. **Clone the Repository:**

Begin by cloning the repository to your local machine using Git:

.. code-block:: bash

    git clone https://github.com/qc-lab/QHyper.git

2. **Create a Virtual Environment:**

We recommend using a virtual environment to isolate your project's dependencies. Create and activate a virtual environment:

.. code-block:: bash

    python -m venv venv
    source venv/bin/activate


3. **Install Production Requirements:**

Install the project's production dependencies using pip:

.. code-block:: bash

    pip install -r requirements/prod.txt

.. note::

    ``requirements/docs.txt`` is not required for general contributions unless you plan to build the documentation locally.

4. **Development Environment Setup Complete:**

Your development environment is now set up and ready.


Updating Documentation
----------------------

If you want to update the project's documentation, follow these steps:

1. **Development Environment Setup:**

Install additonal dependencies for building the documentation:

.. code-block:: bash

   pip install -r requirements/docs.txt

2. **Modify Documentation Files:**

Make the necessary changes to the documentation files located in the ``docs/source`` directory.

3. **Build documentation locally:**

Go into ``docs`` dir:

.. code-block:: bash

   cd docs

And run following command to build documentation:

.. code-block:: bash

   make html

Make sure that there are no errors. You will able to view the documentation in your browser by opening the ``docs/build/html/index.html`` file.

.. note::
    You may want to remove docs/build before running this commands to ensure that you're building the documentation from scratch.

4. **Documentation Update Complete:**

Your documentation changes are ready for submission.



Contribution Rules
------------------

When contributing to this project, please adhere to the following rules:

1. **Use MyPy Typing:**

Ensure that you use type hints following MyPy conventions to enhance code readability and maintainability.

2. **Limit Line Length:**

Keep lines of code and comments to a maximum of 80 characters in length to ensure code readability.

3. **Test Before Creating a Pull Request:**

Before creating a pull request, make sure that all tests pass without errors.

Submitting Your Contribution
----------------------------

When you're ready to submit your contribution, please follow these steps:

1. **Create a Branch:**

Create a new branch with a descriptive name for your contribution. This makes it easier for reviewers to understand the purpose of your changes:

.. code-block:: bash

   git checkout -b your-branch-name

2. **Commit Your Changes:**

Make your changes, commit them, and provide a clear and concise commit message that describes your modifications:

.. code-block:: bash

   git add .
   git commit -m "Your descriptive commit message"

3. **Push Your Branch:**

Push your branch to the remote repository:

.. code-block:: bash

    git push origin your-branch-name

4. **Create a Pull Request (PR):**

Go to the project's repository on GitHub and create a pull request. Ensure that you provide a detailed description of your changes and any related issues.

5. **Review and Collaborate:**

Collaborate with reviewers to address feedback and make any necessary improvements to your contribution.

6. **Merge Your Pull Request:**

Once your pull request has been reviewed and approved, it will be merged into the main project branch.

7. **Thank You:**

Congratulations on your contribution! Thank you for helping improve the project.

By following these guidelines, you'll help ensure a smooth contribution process and maintain the quality of the project. Your contributions are valuable, and we appreciate your efforts to make this project better!
