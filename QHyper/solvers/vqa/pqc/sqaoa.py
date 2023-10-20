from dataclasses import dataclass
import pennylane as qml
from pennylane import numpy as np
from scipy.sparse import csr_matrix

import numpy.typing as npt
from typing import Any, Callable, cast, Optional

from QHyper.problems.base import Problem

from QHyper.solvers.vqa.pqc.base import PQC
from QHyper.solvers.converter import QUBO, Converter

from .mixers import MIXERS_BY_NAME


@dataclass
class SQAOA(PQC):
    layers: int = 3
    backend: str = "default.qubit"
    mixer: str = 'pl_x_mixer'
    offset=0;
    def _create_cost_operator(self, qubo: QUBO) -> qml.Hamiltonian:
        #print(qubo)
        result = qml.Identity(0)-qml.Identity(0)
        for variables, coeff in qubo.items():
            if not variables:
                self.offset=coeff
                continue
            tmp = coeff * (
                0.5 * qml.Identity(str(variables[0]))
                - 0.5 * qml.PauliZ(str(variables[0]))
            )
            if len(variables) == 2 and variables[0] != variables[1]:
                tmp = tmp @ (
                    0.5 * qml.Identity(str(variables[1]))
                    - 0.5 * qml.PauliZ(str(variables[1]))
                )
            result += tmp
       # print(result,"\n")
        #print("qubo offset",self.offset,"\n")
        return result
    
   
 
    def _hadamard_layer(self, problem: Problem) -> None:
        for i in problem.variables:
            qml.Hadamard(str(i))

    def _create_mixing_hamiltonian(self, problem: Problem) -> qml.Hamiltonian:
        if self.mixer not in MIXERS_BY_NAME:
            raise Exception(f"Unknown {self.mixer} mixer")
        return MIXERS_BY_NAME[self.mixer]([str(v) for v in problem.variables])

    def _circuit(self, problem: Problem, params,
                 cost_operator: qml.Hamiltonian) -> None:

        def qaoa_layer(gamma: list[float], beta: list[float]) -> None:
            qml.qaoa.cost_layer(gamma, cost_operator)
            qml.qaoa.mixer_layer(
                beta, self._create_mixing_hamiltonian(problem))

        self._hadamard_layer(problem)
        qml.layer(qaoa_layer, self.layers, params[0], params[1])

    def get_expval_circuit(self, problem: Problem, weights: list[float]
                           ):
        qubo = Converter.create_qubo(problem, weights)
        cost_operator = self._create_cost_operator(qubo)

        @qml.qnode(self.dev)
        def expval_circuit(params: npt.NDArray[np.float64]):
            self._circuit(problem, params, cost_operator)
            #return cast(float, qml.expval(
             #   cost_operator
                # self._create_weight_free_hamiltonian(problem)
            #))
            return qml.expval(
                cost_operator
                # self._create_weight_free_hamiltonian(problem)
            )

        return  expval_circuit

    def run_opt(
        self,
        problem: Problem,
        opt_args: npt.NDArray[np.float64],
        hyper_args: npt.NDArray[np.float64],
        print_results: bool = False
    ):   
            
        self.dev = qml.device(
           self.backend, wires=[str(x) for x in problem.variables])
        self.get_expval_circuit(problem, list(hyper_args))(
           opt_args.reshape(2, -1))
       
        qubo = Converter.create_qubo(problem, list(hyper_args))
        
        
        cost_operator = self._create_cost_operator(qubo)
        bits=9
        def get_score2(result):
            x = np.array(list(np.binary_repr(result,bits)),dtype=int)
           # print(x)
           # return -self.offset+2 * x[0] + 5 * x[1]+ x[0] * x[1] + 3.5* (x[0] + x[1] -1)*(x[0] + x[1] -1)+1.5*(5*x[0] + 2*x[1] - x[2] - 2*x[3] - 2*x[4])*(5*x[0] + 2*x[1] - x[2] - 2*x[3] - 2*x[4])
            return (
                    -self.offset
                    +37.89186033670198*((x[0] + x[1] + x[2] - 1)*(x[0] + x[1] + x[2] - 1)
                                        +(x[3] + x[4] + x[5] - 1)*(x[3] + x[4] + x[5] - 1)
                                        +(x[6] + x[7] + x[8] - 1)*(x[6] + x[7] + x[8] - 1))
                    +(6.0*x[0] + 8.0*x[1] + 8.0*x[2] + 3.0*x[3] + 4.0*x[4] + 4.0*x[5] + 12.0*x[6] + 16.0*x[7] + 16.0*x[8])
                    +15.536726433137282*(13- (6*x[0] + 2*x[1] + 4*x[2] + 3*x[3] + 1*x[4] + 2*x[5] + 12*x[6] + 4*x[7] + 8*x[8]))  
                    + 38.61604208771982*(13-(6*x[0] + 2*x[1] + 4*x[2] + 3*x[3] + 1*x[4] + 2*x[5] + 12*x[6] + 4*x[7] + 8*x[8]))
                    *(13-(6*x[0] + 2*x[1] + 4*x[2] + 3*x[3] + 1*x[4] + 2*x[5] + 12*x[6] + 4*x[7] + 8*x[8])))
        def check_cost(result):
            x = np.array(list(np.binary_repr(result,bits)),dtype=int)
            return 6.0*x[0] + 8.0*x[1] + 8.0*x[2] + 3.0*x[3] + 4.0*x[4] + 4.0*x[5] + 12.0*x[6] + 16.0*x[7] + 16.0*x[8]
            #return 2 * x[0] + 5 * x[1]+ x[0] * x[1] 
        
        def check_const1(result):
            x = np.array(list(np.binary_repr(result,bits)),dtype=int)
            return x[0] + x[1] + x[2] == 1 and x[3] + x[4] + x[5] == 1 and x[6] + x[7] + x[8] == 1
            
            #return  x[0] + x[1] -1 == 0 
        def check_const2(result):
            x = np.array(list(np.binary_repr(result,bits)),dtype=int)
            return  6*x[0] + 2*x[1] + 4*x[2] + 3*x[3] + 1*x[4] + 2*x[5] + 12*x[6] + 4*x[7] + 8*x[8] <= 13
            #return 5 * x[0] + 2 * x[1] <= 5
        def check_const3(result):
            x = np.array(list(np.binary_repr(result,bits)),dtype=int)
           # print(x)
            #print("czas=",6*x[0] + 2*x[1] + 4*x[2] + 3*x[3] + 1*x[4] + 2*x[5] + 12*x[6] + 4*x[7] + 8*x[8])
          #  print(15.536726433137282*13+3.61604208771982*(-13)*(-13))
            return (+15.536726433137282*(13- (6*x[0] + 2*x[1] + 4*x[2] + 3*x[3] + 1*x[4] + 2*x[5] + 12*x[6] + 4*x[7] + 8*x[8]))  
                    + 38.61604208771982*(13-(6*x[0] + 2*x[1] + 4*x[2] + 3*x[3] + 1*x[4] + 2*x[5] + 12*x[6] + 4*x[7] + 8*x[8]))
                    *(13-(6*x[0] + 2*x[1] + 4*x[2] + 3*x[3] + 1*x[4] + 2*x[5] + 12*x[6] + 4*x[7] + 8*x[8])))
            #return 5*x[0] + 2*x[1] - x[2] - 2*x[3] - 2*x[4] == 0
        #for i in range(np.power(2, bits)):
           #print(format(i, '#0{}b'.format(7)), round(abs(qml.matrix(cost_operator)[i,i]),2))
          # print(round(np.real(qml.matrix(cost_operator)[i,i]),2),"b"+bin(i)[2:].zfill(bits), get_score2(i),check_cost(i),check_const1(i),check_const2(i),check_const3(i))
            
        @qml.qnode(self.dev)
        def expval_circuit(params):
           self._circuit(problem,params,cost_operator) 
           return qml.expval(
               cost_operator)
        
        opt = qml.QNGOptimizer(0.00004)
        params = np.array(opt_args, requires_grad=True)
        for ind in range(50):
            params, cost = opt.step_and_cost(expval_circuit,params)
            print(ind, " ", cost," ", params,"\n")    
            
        return self.get_params_init_format(params, hyper_args)

      

    def get_opt_args(
        self,
        params_init: dict[str, Any],
        args: Optional[npt.NDArray[np.float64]] = None,
        hyper_args: Optional[npt.NDArray[np.float64]] = None
    ) -> npt.NDArray[np.float64]:
        return args if args is not None else np.array(params_init['angles'])

    def get_hopt_args(
        self,
        params_init: dict[str, Any],
        args: Optional[npt.NDArray[np.float64]] = None,
        hyper_args: Optional[npt.NDArray[np.float64]] = None
    ) -> npt.NDArray[np.float64]:
        return (
            hyper_args if hyper_args is not None
            else np.array(params_init['hyper_args'])
        )

    def get_params_init_format(
        self,
        opt_args: npt.NDArray[np.float64],
        hyper_args: npt.NDArray[np.float64]
    ) -> dict[str, Any]:
        return {
            'angles': opt_args,
            'hyper_args': hyper_args,
        }
