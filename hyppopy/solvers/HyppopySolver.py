# Hyppopy - A Hyper-Parameter Optimization Toolbox
#
# Copyright (c) German Cancer Research Center,
# Division of Medical Image Computing.
# All rights reserved.
#
# This software is distributed WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.
#
# See LICENSE

__all__ = ['HyppopySolver']

import abc
import copy
import types
import datetime
import numpy as np
import pandas as pd
from mpi4py import MPI
from hyperopt import Trials
from hyppopy.globals import *
from hyppopy.VisdomViewer import VisdomViewer
from hyppopy.HyppopyProject import HyppopyProject
from hyppopy.BlackboxFunction import BlackboxFunction
from hyppopy.FunctionSimulator import FunctionSimulator
from hyppopy.globals import DEBUGLEVEL

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)

"""
abc - Abstract Base Classes
Purpose: Define and use abstract base classes for interface verification.i
Abstract classes are classes that contain one or more abstract methods. An abstract method is a method that is
declared but contains no implementation. Abstract classes may not be instantiated and require subclasses to provide
implementations for the abstract methods.
Abstract base classes are a form of interface checking more strict than individual hasattr() checks for peculiar 
methods. By defining an abstract base class, a common API can be established for a set of subclasses. This
capability is especially useful in situations where someone less familiar with the source for an application is
going to provide plug-in extensions.
abc works by marking methods of the base class as abstract, and then registering concrete classes as implementations
of the abstract base. If an application or library requires a particular API, issubclass() or isinstance() can be
used to check an object against the abstract class. Use decorators to establish the public API for the class.
"""

class HyppopySolver(object):
    """
    The HyppopySolver class is the base class for all solver add-ons. It defines virtual functions a child class has
    to implement to deal with the front-end communication, orchestrate the optimization process and ensure proper
    process information storing.
    The key idea is that the HyppopySolver class defines an interface to configure and run an object instance of itself
    independently from the concrete solver lib used to optimize in the background. To achieve this goal an add-on
    developer needs to implement the abstract methods 'convert_searchspace', 'execute_solver' and 'loss_function_call'.
    These methods abstract the peculiarities of the solver libs to offer and a simple and consistent parameter space 
    configuration and optimization procedure on the user side. The method 'convert_searchspace' transforms the hyppopy
    parameter space description into the solver-lib specific description. The method loss_function_call is used to
    handle solver-lib specifics of calling the actual blackbox function and execute_solver is executed when the run
    method is invoked und takes care of calling the solver lib solving routine.

    The class HyppopySolver defines an interface to be implemented when writing a custom solver. Each solver derivative
    needs to implement the abstract methods:

    - convert_searchspace
    - execute_solver
    - loss_function_call
    - define_interface

    The dev-user interface consists of the methods:

    - _add_member
    - _add_hyperparameter_signature
    - _check_project

    The end-user interface consists of the methods:

    - run
    - get_results
    - print_best
    - print_timestats
    - start_viewer
    """
    def __init__(self, project=None):
        """
        The constructor accepts a HyppopyProject.

        :param project: [HyppopyProject] project instance, default=None
        """
        self._idx = None                        # current iteration counter
        self._best = None                       # best parameter set
        self._trials = None                     # trials object (hyppopy uses the Trials object from hyperopt)
        self._blackbox = None                   # blackbox function (either a function or a BlackboxFunction instance)
        self._total_duration = None             # keep track of the solver's running time
        self._solver_overhead = None            # store time overhead of the solver, i.e. total time minus time in blackbox
        self._time_per_iteration = None         # mean time per iterration
        self._accumulated_blackbox_time = None  # total time the solver was in the blackbox function
        self._visdom_viewer = None              # visdom viewer instance

        self._child_members = {}                # dict keeping track of settings defined by child solver
        self._hopt_signatures = {}              # dict keeping track of hyperparameter signatures defined by child solver
        self.define_interface()                 # child define interface function is called to define settings and hyperparameter signatures

        if project is not None:                 # Hyppopyproject is set as solver attribute 'project'.
            self.project = project              # 'project' attribute makes functions defined by user at top-level interface accessible.

    @abc.abstractmethod
    def convert_searchspace(self, hyperparameter):
        """
        This function gets the unified hyppopy-like parameterspace description as input and, if necessary, converts it
        into a solver-lib specific format. The function is invoked when run is called and what it returns is passed as 
        searchspace argument to the function execute_solver.

        :param hyperparameter: [dict] nested parameter description dict e.g. {'name': {'domain':'uniform', 'data':[0,1], 'type':'float'}, ...}

        :return: [object] converted hyperparameter space
        """
        raise NotImplementedError('Users must define convert_searchspace to use this class.')

    @abc.abstractmethod
    def execute_solver(self, searchspace):
        """
        This function is called immediately after convert_searchspace and uses the output of the latter as input. Its
        purpose is to call the solver lib's main optimization function.

        :param searchspace: converted hyperparameter space
        """
        raise NotImplementedError('Users must define execute_solver to use this class.')

    @abc.abstractmethod
    def loss_function_call(self, params):
        """
        This function is called within the function loss_function and encapsulates the actual blackbox function call
        in each iteration. The function loss_function takes care of the iteration driving and reporting, but each solver
        lib might need some special treatment between the parameter set selection and the calling of the actual blackbox
        function, e.g. parameter converting.

        :param params: [dict] hyperparameter space sample e.g. {'p1': 0.123, 'p2': 3.87, ...}

        :return: [float] loss
        """
        raise NotImplementedError('Users must define loss_function_call to use this class.')

    @abc.abstractmethod
    def define_interface(self):
        """
        This function is called when HyppopySolver.__init__ function finished. Child classes need to define their
        individual parameters here by calling the _add_member function for each class member variable to be defined.
        Using _add_hyperparameter_signature, the structure of a hyperparameter expected by the solver must be defined.
        Both members and hyperparameter signatures are checked later on, before executing the solver, to ensure the
        settings passed fulfill the solver's needs.
        """
        raise NotImplementedError('Users must define define_interface to use this class.')

    def _add_member(self, name, dtype, value=None, default=None):
        """
        When designing your child solver class you need to implement the define_interface abstract method where you can
        call _add_member to define custom solver options that are automatically converted to class attributes.

        :param name: [str] option name
        :param dtype: [type] option data type
        :param value: [object] option value
        :param default: [object] option default value
        """
        assert isinstance(name, str), "Precondition violation, name needs to be of type str, got {}.".format(type(name))
        if value is not None:
            assert isinstance(value, dtype), "Precondition violation, value does not match dtype condition!"
        if default is not None:
            assert isinstance(default, dtype), "Precondition violation, default does not match dtype condition!"
        setattr(self, name, value)
        self._child_members[name] = {"type": dtype, "value": value, "default": default}

    def _add_hyperparameter_signature(self, name, dtype, options=None):
        """
        When designing your child solver class you need to implement the define_interface abstract method where you can
        call _add_hyperparameter_signature to define a hyperparamter signature which is automatically checked for
        consistency while solver execution.

        :param name: [str] hyperparameter name
        :param dtype: [type] hyperparameter data type
        :param options: [list] list of possible values the hp can be set, if None no option check is done
        """
        assert isinstance(name, str), "Precondition violation, name needs to be of type str, got {}.".format(type(name))
        self._hopt_signatures[name] = {"type": dtype, "options": options}

    def _check_project(self):
        """
        The function checks the members and hyperparameter signatures read from the project instance to be consistent
            with the members and signatures defined in the child class via define_interface.
        """
        assert isinstance(self.project, HyppopyProject), "Invalid project instance, either not set or setting failed!"

        # check hyperparameter signatures
        for name, param in self.project.hyperparameter.items():
            for sig, settings in self._hopt_signatures.items():
                if sig not in param.keys():
                    msg = "Missing hyperparameter signature {}!".format(sig)
                    LOG.error(msg)
                    raise LookupError(msg)
                else:
                    if not isinstance(param[sig], settings["type"]):
                        msg = "Hyperparameter signature type mismatch, expected type {} got {}!".format(settings["type"], param[sig])
                        LOG.error(msg)
                        raise TypeError(msg)
                    if settings["options"] is not None:
                        if param[sig] not in settings["options"]:
                            msg = "Wrong signature value, {} not found in signature options!".format(param[sig])
                            LOG.error(msg)
                            raise LookupError(msg)

        # check child members
        for name in self._child_members.keys():
            if name not in self.project.__dict__.keys():
                msg = "Missing settings field {}!".format(name)
                LOG.error(msg)
                raise LookupError(msg)
            self.__dict__[name] = self.project.settings[name]

    def __compute_time_statistics(self):
        """
        Evaluates all time statistic values available
        """
        dts = []
        for trial in self._trials.trials:
            if 'book_time' in trial.keys() and 'refresh_time' in trial.keys():
                dt = trial['refresh_time'] - trial['book_time']
                dts.append(dt.total_seconds())
        self._time_per_iteration = np.mean(dts) * 1e3
        self._accumulated_blackbox_time = np.sum(dts) * 1e3
        tmp = self.total_duration - self._accumulated_blackbox_time
        self._solver_overhead = int(np.round(100.0 / (self.total_duration + 1e-12) * tmp))

    def loss_function(self, **params):
        """
        This function is called each iteration with a selected parameter set. The parameter set selection is driven by
        the solver lib itself. The purpose of this function is to take care of the iteration reporting and the calling
        of the callback_func is available. As a developer you might want to overwrite this function completely (e.g.
        HyperoptSolver) but then you need to take care of iteration reporting by yourself. The alternative is to only
        implement loss_function_call (e.g. OptunitySolver).

        :param params: [dict] hyperparameter space sample e.g. {'p1': 0.123, 'p2': 3.87, ...}

        :return: [float] loss
        """
        self._idx += 1
        vals = {}
        idx = {}
        for key, value in params.items():
            vals[key] = [value]
            idx[key] = [self._idx]
        trial = {'tid': self._idx,
                 'result': {'loss': None, 'status': 'ok'},
                 'misc': {
                     'tid': self._idx,
                     'idxs': idx,
                     'vals': vals
                 },
                 'book_time': datetime.datetime.now(),
                 'refresh_time': None
                 }
        try:
            loss = self.loss_function_call(params)
            trial['result']['loss'] = loss
            trial['result']['status'] = 'ok'
            if loss is np.nan:
                trial['result']['status'] = 'failed'
        except Exception as e:
            LOG.error("computing loss failed due to:\n {}".format(e))
            loss = np.nan
            trial['result']['loss'] = np.nan
            trial['result']['status'] = 'failed'
        trial['refresh_time'] = datetime.datetime.now()
        self._trials.trials.append(trial)
        cbd = copy.deepcopy(params)
        cbd['iterations'] = self._idx
        cbd['loss'] = loss
        cbd['status'] = trial['result']['status']
        cbd['book_time'] = trial['book_time']
        cbd['refresh_time'] = trial['refresh_time']
        if isinstance(self.blackbox, BlackboxFunction) and self.blackbox.callback_func is not None:
            self.blackbox.callback_func(**cbd)
        if self._visdom_viewer is not None:
            self._visdom_viewer.update(cbd)
        return loss

    def run(self, print_stats=True):
        """
        This function starts the optimization process.

        :param print_stats: [bool] en- or disable console output
        """
        self._idx = 0
        self.trials = Trials()

        start_time = datetime.datetime.now()
        try:
            search_space = self.convert_searchspace(self.project.hyperparameter)
        except Exception as e:
            msg = "Failed to convert searchspace, error: {}".format(e)
            LOG.error(msg)
            raise AssertionError(msg)
        try:
            self.execute_solver(search_space)
        except Exception as e:
            msg = "Failed to execute solver, error: {}".format(e)
            LOG.error(msg)
            raise AssertionError(msg)
        end_time = datetime.datetime.now()
        dt = end_time - start_time
        days = divmod(dt.total_seconds(), 86400)
        hours = divmod(days[1], 3600)
        minutes = divmod(hours[1], 60)
        seconds = divmod(minutes[1], 1)
        milliseconds = divmod(seconds[1], 0.001)
        self._total_duration = [int(days[0]), int(hours[0]), int(minutes[0]), int(seconds[0]), int(milliseconds[0])]
        if print_stats:
            self.print_best()
            self.print_timestats()

    def get_results(self):
        """
        This function returns a complete optimization history as pandas DataFrame (data manipulation and analysis) and 
        a dict with the optimal parameter set.

        :return: [DataFrame], [dict] history and optimal parameter set
        """
        assert isinstance(self.trials, Trials), "Precondition violation, wrong trials type! Maybe solver was not yet executed?"
        results = {'duration': [], 'losses': [], 'status': []}
        pset = self.trials.trials[0]['misc']['vals']
        for p in pset.keys():
            results[p] = []

        for n, trial in enumerate(self.trials.trials):
            t1 = trial['book_time']
            t2 = trial['refresh_time']
            results['duration'].append((t2 - t1).microseconds / 1000.0)
            results['losses'].append(trial['result']['loss'])
            results['status'].append(trial['result']['status'] == 'ok')
            losses = np.array(results['losses'])
            results['losses'] = list(losses)
            pset = trial['misc']['vals']
            for p in pset.items():
                results[p[0]].append(p[1][0])
        return pd.DataFrame.from_dict(results), self.best

    def print_best(self):
        """
        Optimization result console output printing.
        """
        print("\n")
        print("#" * 40)
        print("###       Best Parameter Choice      ###")
        print("#" * 40)
        for name, value in self.best.items():
            print(" - {}\t:\t{}".format(name, value))
        print("\n - number of iterations\t:\t{}".format(self.trials.trials[-1]['tid']+1))
        print(" - total time\t:\t{}d:{}h:{}m:{}s:{}ms".format(self._total_duration[0],
                                                              self._total_duration[1],
                                                              self._total_duration[2],
                                                              self._total_duration[3],
                                                              self._total_duration[4]))
        print("#" * 40)

    def print_timestats(self):
        """
        Time statistic console output printing.
        """
        print("\n")
        print("#" * 40)
        print("###        Timing Statistics        ###")
        print("#" * 40)
        print(" - per iteration: {}ms".format(int(self.time_per_iteration*1e4)/10000))
        print(" - total time: {}d:{}h:{}m:{}s:{}ms".format(self._total_duration[0],
                                                           self._total_duration[1],
                                                           self._total_duration[2],
                                                           self._total_duration[3],
                                                           self._total_duration[4]))
        print("#" * 40)
        print(" - solver overhead: {}%".format(self.solver_overhead))

    def start_viewer(self, port=8097, server="http://localhost"):
        """
        Starts the visdom viewer.

        :param port: [int] port number, default: 8097
        :param server:  [str] server name, default: http://localhost
        """
        try:
            self._visdom_viewer = VisdomViewer(self._project, port, server)
        except Exception as e:
            import warnings
            warnings.warn("Failed starting VisdomViewer. Is the server running? If not start it via $visdom")
            LOG.error("Failed starting VisdomViewer: {}".format(e))
            self._visdom_viewer = None

    @property
    def project(self):
        """
        HyppopyProject instance

        :return: [HyppopyProject] project instance
        """
        return self._project

    @project.setter
    def project(self, value):
        """
        Set HyppopyProject instance

        :param value: [HyppopyProject] project instance
        """
        if isinstance(value, dict):
            self._project = HyppopyProject(value)
        elif isinstance(value, HyppopyProject):
            self._project = value
        else:
            msg = "Input error, project_manager of type: {} not allowed!".format(type(value))
            LOG.error(msg)
            raise TypeError(msg)
        self._check_project()

    @property
    def blackbox(self):
        """
        Get the BlackboxFunction object.

        :return: [object] BlackboxFunction instance or function
        """
        return self._blackbox

    @blackbox.setter
    def blackbox(self, value):
        """
        Set the BlackboxFunction wrapper class encapsulating the loss function or a function accepting a hyperparameter set
        and returning a float.

        :return: [object] pointer to blackbox_func
        """
        if isinstance(value, types.FunctionType) or isinstance(value, BlackboxFunction) or isinstance(value, FunctionSimulator):
            self._blackbox = value
        else:
            self._blackbox = None
            msg = "Input error, blackbox of type: {} not allowed!".format(type(value))
            LOG.error(msg)
            raise TypeError(msg)

    @property
    def best(self):
        """
        Returns best parameter set.

        :return: [dict] best parameter set
        """
        return self._best

    @best.setter
    def best(self, value):
        """
        Set the best parameter set.

        :param value: [dict] best parameter set

        """
        if not isinstance(value, dict):
            msg = "Input error, best of type: {} not allowed!".format(type(value))
            LOG.error(msg)
            raise TypeError(msg)
        self._best = value

    @property
    def trials(self):
        """
        Get the Trials instance.

        :return: [object] Trials instance
        """
        return self._trials

    @trials.setter
    def trials(self, value):
        """
        Set the Trials object.

        :param value: [object] Trials instance
        """
        self._trials = value

    @property
    def total_duration(self):
        """
        Get total computation duration.

        :return: [float] total computation time
        """
        return (self._total_duration[0]*86400 + self._total_duration[1] * 3600 + self._total_duration[2] * 60 + self._total_duration[3]) * 1000 + self._total_duration[4]

    @property
    def solver_overhead(self):
        """
        Get the solver overhead, this is the total time minus the duration of the blackbox function calls.

        :return: [float] solver overhead duration
        """
        if self._solver_overhead is None:
            self.__compute_time_statistics()
        return self._solver_overhead

    @property
    def time_per_iteration(self):
        """
        Get the mean duration per iteration.

        :return: [float] time per iteration
        """
        if self._time_per_iteration is None:
            self.__compute_time_statistics()
        return self._time_per_iteration

    @property
    def accumulated_blackbox_time(self):
        """
        Get the summed blackbox function computation time.

        :return: [float] blackbox function computation time
        """
        if self._accumulated_blackbox_time is None:
            self.__compute_time_statistics()
        return self._accumulated_blackbox_time
