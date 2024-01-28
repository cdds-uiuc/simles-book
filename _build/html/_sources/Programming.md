# Programming

## Python
The class assumes basic knowledge of Python. It will also make use of a number of libraries for scientific computing. The most commonly used packages are <code>numpy</code>, <code>pandas</code>, <code>scipy</code>,<code>scikit-learn</code>, and <code>matplotlib</code>. We will also make ocassional use of <code>xarray</code>, <code>cartopy</code>, <code>statsmodel</code>, <code>seaborn</code> and many others. 
 The course notebooks will give you ample examples of how to use these libraries, so it is not necessary to be familiar with them beforehand. At the same time, this is not a scientific computing class, so it is assumed that you can read and reproduce scientific computing code, and are able to look up documentations and examples by yourself.

If your goal is learning the  basics of handling datasets (including large datasets)  I highly recommend the more programming focused ATMS 517 / GEOL 517 - Data Science for Geosciences. 

Below I list some useful resources, and outline what you will need to know for the course. 

## Resources
There are two main resources I recommend, in the form of online Jupyterbooks, developed and updated by some of the leading developers of Python for Earth Sciences (primarily Ryan Abernathey and Brian Rose).
1. **[Project Pythia Foundations](https://foundations.projectpythia.org/landing-page.html)**
2. **[Earth And Environmental Data Science(EEDS)](https://earth-env-data-science.github.io/intro.html)**

Project Pythia does a good job at collating a number of other great tutorials and resources. [Project Pythia Cookbooks](https://cookbooks.projectpythia.org/) are a set of well-currated tutorials, though with a bias towards atmosphere and ocean topics. A broader set of resources can be found under [Project Pythia's Resource Page](https://projectpythia.org/resource-gallery.html)

I have also made great use of [Software Carpentry](https://software-carpentry.org/lessons/). 

## What you need to know:
### Intro to  Python
If you don't know basic Python (i.e. opening Python, printing, assigning variables, accessing values in a list, etc) I recommend going through the (1) *Foundational Skills* section of Project Pythia's Foundations and (2) *The Cory Python Language* in Earth and Environmental Data Science. If you find this is not enough, go through the [Programming with Python](https://swcarpentry.github.io/python-novice-inflammation/) and [Programming and Plotting in Python](https://swcarpentry.github.io/python-novice-gapminder) lessons from Software Carpentry.  

### Basic Python libraries
The basic libraries we will use are [numpy](https://numpy.org/) and [matplotlib.](https://matplotlib.org/) We will be using these very early in the course, so if you find you don't know them (or would like a refresher) make use of Pythia and EEDS. 

I will introduce a few other libraries, most importantly [SciPy](https://scipy.org/) and [scikit-learn](https://scikit-learn.org/stable/) These are essentially numpy add-ons, and fairly easy to use if you you’ve used numpy.

 
### Advanced: 

**Libraries:** We will make use of [pandas](https://pandas.pydata.org/) and [xarray](https://docs.xarray.dev/en/stable/) and [cartopy](https://pypi.org/project/Cartopy/) [seaborn](https://www.google.com/search?q=searborn&oq=searborn&gs_lcrp=EgZjaHJvbWUyBggAEEUYOdIBCDExMTdqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8) and [statsmodel](https://www.statsmodels.org/stable/index.html) although it will be a few weeks until we do so. Pythia and EEDS have good tutorials for these. I will also provide usable totorials for you to start from. You should not need advanced knowledge of these libraries for this course, although they are generally useful. 

**Environments & Github:** For your final projects you will need to make your code reproducible which means using a reproducible environment (via a .yml file) and using github. EEDS provides a usable introduction to Environments, while both Pythia and EEDS provide introductions to git and Github.
