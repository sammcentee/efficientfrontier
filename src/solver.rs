//! Convex programs for the Clarabel interior-point solver.
//!
//! The Python release used CVXPY with Clarabel. This module builds the same
//! conic form directly: minimize `x'Px / 2 + q'x` subject to linear equalities,
//! linear inequalities and Euclidean-norm bounds. Statuses keep CVXPY's names,
//! and a solver failure keeps CVXPY's message, so errors read the same.

use clarabel::algebra::CscMatrix;
use clarabel::solver::{DefaultSettingsBuilder, DefaultSolver, IPSolver, SolverStatus, SupportedConeT};
use nalgebra::DMatrix;

use crate::error::{Error, Result};

/// The solver failure text that CVXPY gave for Clarabel.
pub const SOLVER_FAILURE: &str =
    "Solver 'CLARABEL' failed. Try another solver, or solve with verbose=True for more information.";

/// The outcome of a solve, named as CVXPY names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Optimal,
    OptimalInaccurate,
    Infeasible,
    InfeasibleInaccurate,
    Unbounded,
    UnboundedInaccurate,
    UserLimit,
}

impl Status {
    /// CVXPY's status text, such as `infeasible`.
    pub fn name(self) -> &'static str {
        match self {
            Status::Optimal => "optimal",
            Status::OptimalInaccurate => "optimal_inaccurate",
            Status::Infeasible => "infeasible",
            Status::InfeasibleInaccurate => "infeasible_inaccurate",
            Status::Unbounded => "unbounded",
            Status::UnboundedInaccurate => "unbounded_inaccurate",
            Status::UserLimit => "user_limit",
        }
    }

    pub fn is_optimal(self) -> bool {
        matches!(self, Status::Optimal | Status::OptimalInaccurate)
    }

    pub fn is_infeasible(self) -> bool {
        matches!(self, Status::Infeasible | Status::InfeasibleInaccurate)
    }
}

/// A solve result. `x` is meaningful only when the status is optimal.
#[derive(Clone, Debug)]
pub struct Solution {
    pub status: Status,
    pub x: Vec<f64>,
}

/// A sparse row: `(variable, coefficient)` pairs.
pub type Row = Vec<(usize, f64)>;

/// One convex program over `n` variables.
#[derive(Clone, Debug)]
pub struct Program {
    n: usize,
    quadratic: Option<DMatrix<f64>>,
    linear: Vec<f64>,
    equalities: Vec<(Row, f64)>,
    inequalities: Vec<(Row, f64)>,
    norm_bounds: Vec<(DMatrix<f64>, f64)>,
}

impl Program {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            quadratic: None,
            linear: vec![0.0; n],
            equalities: Vec::new(),
            inequalities: Vec::new(),
            norm_bounds: Vec::new(),
        }
    }

    pub fn variables(&self) -> usize {
        self.n
    }

    /// Add `x'Px / 2` to the objective. `p` must be symmetric positive semidefinite.
    pub fn quadratic(mut self, p: DMatrix<f64>) -> Self {
        assert_eq!((p.nrows(), p.ncols()), (self.n, self.n), "P must be n by n");
        self.quadratic = Some(p);
        self
    }

    /// Minimize `q'x`.
    pub fn minimize(mut self, q: Vec<f64>) -> Self {
        assert_eq!(q.len(), self.n, "q must have n entries");
        self.linear = q;
        self
    }

    /// Maximize `c'x`.
    pub fn maximize(self, c: &[f64]) -> Self {
        let q = c.iter().map(|value| -value).collect();
        self.minimize(q)
    }

    /// `a'x == rhs`.
    pub fn equal(mut self, row: Row, rhs: f64) -> Self {
        self.equalities.push((row, rhs));
        self
    }

    /// `a'x <= rhs`.
    pub fn at_most(mut self, row: Row, rhs: f64) -> Self {
        self.inequalities.push((row, rhs));
        self
    }

    /// `a'x >= rhs`.
    pub fn at_least(self, row: Row, rhs: f64) -> Self {
        let negated = row.into_iter().map(|(j, value)| (j, -value)).collect();
        self.at_most(negated, -rhs)
    }

    /// `||F x||_2 <= bound` for a factor with `n` columns.
    pub fn norm_at_most(mut self, factor: DMatrix<f64>, bound: f64) -> Self {
        assert_eq!(factor.ncols(), self.n, "the norm factor must have n columns");
        self.norm_bounds.push((factor, bound));
        self
    }

    /// Solve with the tolerances of the Python release. A solver failure is an
    /// error. Every other outcome, including infeasibility, returns a status.
    pub fn solve(&self) -> Result<Solution> {
        let p = upper_triangle(self.quadratic.as_ref(), self.n);
        let (a, b, cones) = self.constraints();
        let settings = DefaultSettingsBuilder::default()
            .verbose(false)
            .tol_gap_abs(1e-9)
            .tol_gap_rel(1e-9)
            .tol_feas(1e-9)
            .max_iter(200)
            .build()
            .map_err(|_| failure())?;
        let mut solver = DefaultSolver::new(&p, &self.linear, &a, &b, &cones, settings).map_err(|_| failure())?;
        solver.solve();
        let status = match solver.solution.status {
            SolverStatus::Solved => Status::Optimal,
            SolverStatus::AlmostSolved => Status::OptimalInaccurate,
            SolverStatus::PrimalInfeasible => Status::Infeasible,
            SolverStatus::AlmostPrimalInfeasible => Status::InfeasibleInaccurate,
            SolverStatus::DualInfeasible => Status::Unbounded,
            SolverStatus::AlmostDualInfeasible => Status::UnboundedInaccurate,
            SolverStatus::MaxIterations | SolverStatus::MaxTime => Status::UserLimit,
            _ => return Err(failure()),
        };
        Ok(Solution { status, x: solver.solution.x.clone() })
    }

    fn constraints(&self) -> (CscMatrix<f64>, Vec<f64>, Vec<SupportedConeT<f64>>) {
        let mut rows = Vec::new();
        let mut columns = Vec::new();
        let mut values = Vec::new();
        let mut b = Vec::new();
        let mut cones = Vec::new();
        let mut push_row = |row: &Row, rhs: f64, b: &mut Vec<f64>| {
            let index = b.len();
            for (j, value) in row {
                assert!(*j < self.n, "constraint variable out of range");
                if *value != 0.0 {
                    rows.push(index);
                    columns.push(*j);
                    values.push(*value);
                }
            }
            b.push(rhs);
        };
        for (row, rhs) in &self.equalities {
            push_row(row, *rhs, &mut b);
        }
        if !self.equalities.is_empty() {
            cones.push(SupportedConeT::ZeroConeT(self.equalities.len()));
        }
        for (row, rhs) in &self.inequalities {
            push_row(row, *rhs, &mut b);
        }
        if !self.inequalities.is_empty() {
            cones.push(SupportedConeT::NonnegativeConeT(self.inequalities.len()));
        }
        for (factor, bound) in &self.norm_bounds {
            // s = b - Ax with s0 = bound and s[1:] = F x.
            push_row(&Vec::new(), *bound, &mut b);
            for i in 0..factor.nrows() {
                let row: Row = (0..self.n).map(|j| (j, -factor[(i, j)])).collect();
                push_row(&row, 0.0, &mut b);
            }
            cones.push(SupportedConeT::SecondOrderConeT(factor.nrows() + 1));
        }
        let a = CscMatrix::new_from_triplets(b.len(), self.n, rows, columns, values);
        (a, b, cones)
    }
}

fn failure() -> Error {
    Error::Runtime(SOLVER_FAILURE.to_string())
}

/// The upper triangle of a dense symmetric matrix in compressed columns.
fn upper_triangle(matrix: Option<&DMatrix<f64>>, n: usize) -> CscMatrix<f64> {
    let mut colptr = vec![0];
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();
    for j in 0..n {
        if let Some(matrix) = matrix {
            for i in 0..=j {
                let value = matrix[(i, j)];
                if value != 0.0 {
                    rowval.push(i);
                    nzval.push(value);
                }
            }
        }
        colptr.push(rowval.len());
    }
    CscMatrix::new(n, n, colptr, rowval, nzval)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadratic_program_finds_the_minimum_variance_mix() {
        // Two independent assets with variances 1 and 4: weights 0.8 and 0.2.
        let p = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 4.0]);
        let solution = Program::new(2)
            .quadratic(p)
            .equal(vec![(0, 1.0), (1, 1.0)], 1.0)
            .at_least(vec![(0, 1.0)], 0.0)
            .at_least(vec![(1, 1.0)], 0.0)
            .solve()
            .unwrap();
        assert_eq!(solution.status, Status::Optimal);
        assert!((solution.x[0] - 0.8).abs() < 1e-7);
        assert!((solution.x[1] - 0.2).abs() < 1e-7);
    }

    #[test]
    fn norm_bound_limits_risk_and_reports_infeasibility() {
        let factor = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]);
        let program = |bound: f64| {
            Program::new(2)
                .maximize(&[0.0, 1.0])
                .equal(vec![(0, 1.0), (1, 1.0)], 1.0)
                .at_least(vec![(0, 1.0)], 0.0)
                .at_least(vec![(1, 1.0)], 0.0)
                .norm_at_most(factor.clone(), bound)
        };
        let solution = program(1.0).solve().unwrap();
        assert_eq!(solution.status, Status::Optimal);
        // (1-b)^2 + 4b^2 = 1 gives b = 0.4.
        assert!((solution.x[1] - 0.4).abs() < 1e-7);
        assert!(program(0.5).solve().unwrap().status.is_infeasible());
    }
}
