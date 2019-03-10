use indicatif::{ProgressBar, ProgressStyle};
use ndarray::Array;
use ndarray_rand::RandomExt;
use rand::distributions::{Distribution, Normal};
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use std::fmt;

pub type Input = ndarray::Array2<f64>;
pub type Output = ndarray::Array2<f64>;
pub type LayerBiases = ndarray::Array2<f64>;
pub type LayerWeights = ndarray::Array2<f64>;
pub type LayerActivations = ndarray::Array2<f64>;
pub type LayerDeltas = ndarray::Array2<f64>;

#[derive(PartialEq, Eq, Debug, Clone, Serialize, Deserialize)]
pub enum Label {
    Zero,
    One,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
}

impl Label {
    /// Convert the label into an ideal activation, without taking ownership
    pub fn to_arr(&self) -> LayerActivations {
        let mut arr_rep = vec![0.; 10];
        match self {
            Label::Zero => arr_rep[0] = 1.,
            Label::One => arr_rep[1] = 1.,
            Label::Two => arr_rep[2] = 1.,
            Label::Three => arr_rep[3] = 1.,
            Label::Four => arr_rep[4] = 1.,
            Label::Five => arr_rep[5] = 1.,
            Label::Six => arr_rep[6] = 1.,
            Label::Seven => arr_rep[7] = 1.,
            Label::Eight => arr_rep[8] = 1.,
            Label::Nine => arr_rep[9] = 1.,
        }

        Array::from_vec(arr_rep).into_shape((10, 1)).unwrap()
    }

    #[allow(dead_code)]
    pub fn to_num(label: &Label) -> u32 {
        match label {
            &Label::Zero => 0,
            &Label::One => 1,
            &Label::Two => 2,
            &Label::Three => 3,
            &Label::Four => 4,
            &Label::Five => 5,
            &Label::Six => 6,
            &Label::Seven => 7,
            &Label::Eight => 8,
            &Label::Nine => 9,
        }
    }

    pub fn from_num(num: u32) -> Label {
        match num {
            0 => Label::Zero,
            1 => Label::One,
            2 => Label::Two,
            3 => Label::Three,
            4 => Label::Four,
            5 => Label::Five,
            6 => Label::Six,
            7 => Label::Seven,
            8 => Label::Eight,
            9 => Label::Nine,
            _ => panic!("Trying to create label from out-of-range number"),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct Network {
    pub layer_sizes: Vec<u32>,
    pub biases: Vec<LayerBiases>,
    pub weights: Vec<LayerWeights>,
}

impl fmt::Display for Network {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Network({:?})", self.layer_sizes)
    }
}

impl Network {
    pub fn new(layer_sizes: Vec<u32>) -> Network {
        // Sample everything from N(0, 1)
        let normal = Normal::new(0., 1.);

        let biases = Self::init_biases(&layer_sizes, &normal);
        let weights = Self::init_weights(&layer_sizes, &normal);

        Network {
            layer_sizes,
            biases,
            weights,
        }
    }

    fn feedforward_get_all_activations(&self, inp: Input) -> (Vec<LayerActivations>, Vec<LayerActivations>) {
        let mut raw_activations = Vec::with_capacity(self.layer_sizes.len() - 1);
        let mut activations = Vec::with_capacity(self.layer_sizes.len());
        activations.push(inp.clone());

        let mut out = inp;
        let mut raw_out;


        for (biases, weights) in self.biases.iter().zip(&self.weights).take(self.biases.len() - 1) {
            raw_out = weights.dot(&out) + biases;
            out = raw_out.map(|x| Self::sigmoid(*x));
            activations.push(out.clone());
            raw_activations.push(raw_out.clone());
        }

        // Apply a softmax to the last layer
        let lw = &self.weights[self.weights.len() - 1];
        let lb = &self.biases[self.biases.len() - 1];
        raw_out = lw.dot(&out) + lb;
        raw_activations.push(raw_out.clone());
        let softmax_denom = raw_out.map(|x| x.exp()).sum();
        activations.push(raw_out.map(|x| x.exp()) / softmax_denom);

        (activations, raw_activations)
    }

    pub fn feedforward(&self, inp: Input) -> Output {
        self.feedforward_get_all_activations(inp).0.last().unwrap().clone()
    }

    pub fn predict(&self, inp: Input) -> Label {
        let out = self.feedforward(inp);

        let mut prediction = 0;
        let mut max_score = 0.;
        for i in 0..10 {
            if out[[i, 0]] > max_score {
                max_score = out[[i, 0]];
                prediction = i;
            }
        }

        Label::from_num(prediction as u32)
    }

    /// Generate network biases (used during initialization) by sampling from the
    /// given distribution.
    fn init_biases<D>(layer_sizes: &Vec<u32>, distribution: &D) -> Vec<LayerBiases>
    where
        D: Distribution<f64>,
    {
        let mut biases = Vec::with_capacity(layer_sizes.len() - 1);

        for layer in layer_sizes.iter().skip(1) {
            biases.push(Array::random((*layer as usize, 1), distribution));
        }

        biases
    }

    /// Compute the deltas from backpropagating a single training example
    fn backprop(&self, input: &Input, label: &Label) -> (Vec<LayerBiases>, Vec<LayerWeights>) {
        let mut nabla_b: Vec<LayerBiases> = Vec::with_capacity(self.layer_sizes.len() - 1);
        let mut nabla_w: Vec<LayerWeights> = Vec::with_capacity(self.layer_sizes.len() - 1);

        for layer_biases in &self.biases {
            nabla_b.push(Array::zeros(layer_biases.raw_dim()));
        }

        for layer_weights in &self.weights {
            nabla_w.push(Array::zeros(layer_weights.raw_dim()));
        }

        // Store the raw and non-linear activations at each layer, feeding forward
        // the input through the network
        let (activations, zs) = self.feedforward_get_all_activations(input.clone());

        let mut delta = Self::cost_derivative(&activations[activations.len() - 1], label);
        let (nbl, nwl) = (nabla_b.len(), nabla_w.len());
        nabla_b[nbl - 1] = delta.clone();
        nabla_w[nwl - 1] = delta.dot(&activations[activations.len() - 2].t());

        for layer_ind in 2..self.layer_sizes.len() {
            let z = &zs[zs.len() - layer_ind];
            delta = self.weights[self.weights.len() - layer_ind + 1]
                .t()
                .dot(&delta)
                * z.map(|x| Self::sigmoid_derivative(*x));

            nabla_b[nbl - layer_ind] = delta.clone();
            nabla_w[nwl - layer_ind] =
                delta.dot(&activations[activations.len() - layer_ind - 1].t());
        }

        (nabla_b, nabla_w)
    }

    /// The derivative of the cross-entropy cost function
    fn cost_derivative(
        output_activations: &LayerActivations,
        true_activations: &Label,
    ) -> LayerDeltas {
        output_activations - &true_activations.to_arr()
    }

    /// Updates weights and biases by doing gradient descent on a minibatch
    fn update_minibatch(&mut self, minibatch: &mut [(Input, Label)], learning_rate: f64) {
        let per_example_lr = learning_rate / minibatch.len() as f64;

        let mut nabla_b: Vec<LayerBiases> = Vec::with_capacity(self.biases.len());
        let mut nabla_w: Vec<LayerWeights> = Vec::with_capacity(self.weights.len());

        for layer_biases in &self.biases {
            nabla_b.push(Array::zeros(layer_biases.raw_dim()));
        }

        for layer_weights in &self.weights {
            nabla_w.push(Array::zeros(layer_weights.raw_dim()));
        }

        for (input, label) in minibatch {
            let (dnabla_b, dnabla_w) = self.backprop(input, label);

            for (nb, dnb) in nabla_b.iter_mut().zip(&dnabla_b) {
                *nb += dnb;
            }

            for (nw, dnw) in nabla_w.iter_mut().zip(&dnabla_w) {
                *nw += dnw;
            }
        }

        // Update the weights and biases against the direction of the gradient
        for (b, nb) in self.biases.iter_mut().zip(nabla_b) {
            *b -= &(per_example_lr * nb);
        }

        for (w, nw) in self.weights.iter_mut().zip(&nabla_w) {
            *w -= &(per_example_lr * nw);
        }
    }

    /// Train the network using SGD for `num_epochs` epochs, checking accuracy on
    /// `validation_data` if it's provided after each epoch.
    pub fn train(
        &mut self,
        mut training_data: Vec<(Input, Label)>,
        num_epochs: u32,
        minibatch_size: u32,
        learning_rate: f64,
        validation_data: Vec<(Input, Label)>,
    ) {
        let mut rng = thread_rng();
        for epoch in 0..num_epochs {
            println!("Training epoch {} / {}", epoch, num_epochs);
            // Shuffle the data so minibatch slices are randomized in each epoch
            training_data.shuffle(&mut rng);

            let pb = ProgressBar::new(training_data.len() as u64 / minibatch_size as u64);
            pb.set_style(
                ProgressStyle::default_bar().template("[{elapsed_precise}] {wide_bar} ({eta})"),
            );
            // Update each minibatch
            for start_ind in (0..training_data.len()).step_by(minibatch_size as usize) {
                // Avoid going out of bounds if if the minibatch size doesn't divide
                // the length of the input training data
                let upper_ind = std::cmp::min(start_ind + minibatch_size as usize, training_data.len());
                let minibatch = &mut training_data[start_ind..upper_ind as usize];

                self.update_minibatch(minibatch, learning_rate);
                pb.inc(1);
            }
            pb.finish_with_message("Finished epoch");

            // Assess the current performance on the provided holdout set
            let (holdout_correct, holdout_total) = self.evaluate(&validation_data);
            println!(
                "[Training] Epoch {} done. Holdout accuracy: {} / {}",
                epoch, holdout_correct, holdout_total
            );
        }
    }

    fn evaluate(&self, validation_data: &Vec<(Input, Label)>) -> (u32, u32) {
        let mut num_correct = 0;

        for (input, label) in validation_data {
            if self.predict(input.clone()) == *label {
                num_correct += 1;
            }
        }

        (num_correct, validation_data.len() as u32)
    }

    /// Generate network weights (used during initialization) by sampling from the
    /// given distribution.
    fn init_weights<D>(layer_sizes: &Vec<u32>, distribution: &D) -> Vec<LayerBiases>
    where
        D: Distribution<f64>,
    {
        let mut weights = Vec::with_capacity(layer_sizes.len() - 1);

        let num_weights = layer_sizes.len() - 1 as usize;
        for (l1, l2) in layer_sizes
            .iter()
            .take(num_weights)
            .zip(layer_sizes.iter().skip(1))
        {
            weights.push(Array::random((*l2 as usize, *l1 as usize), distribution) / (*l1 as f64).sqrt());
        }

        weights
    }

    /// Sigmoid activation function (mapped over ndarrays)
    pub fn sigmoid(z: f64) -> f64 {
        1. / (1. + (-z).exp())
    }

    /// Derivative of the sigmoid activation function
    fn sigmoid_derivative(z: f64) -> f64 {
        Self::sigmoid(z) * (1. - Self::sigmoid(z))
    }
}

/// Turn a input vector (read in from the CSV) into a ndarray training input
pub fn vector_to_input(inp: Vec<f64>) -> Input {
    assert!(inp.len() == 784);
    Array::from_vec(inp).into_shape((784, 1)).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Dim;

    #[test]
    // Checks that the dimensions are all correct when the network is initialized
    fn network_initializes_correctly() {
        let layer_sizes = vec![784, 30, 10];
        let network = Network::new(layer_sizes.clone());

        assert_eq!(network.layer_sizes, layer_sizes);

        for (layer_biases, layer_size) in network
            .biases
            .iter()
            .zip(network.layer_sizes.iter().skip(1))
        {
            assert_eq!(layer_biases.raw_dim(), Dim([*layer_size as usize, 1]));
        }

        for (i, layer_weights) in network
            .weights
            .iter()
            .take(layer_sizes.len() - 1)
            .enumerate()
        {
            assert_eq!(
                layer_weights.raw_dim(),
                Dim([layer_sizes[i + 1] as usize, layer_sizes[i] as usize])
            );
        }
    }

    #[test]
    // Checks that feedforward computations are working correctly on a super simple examples
    fn feedforward_works() {
        // 1 input 1 output
        let network = Network::new(vec![1, 1]);
        let inp = Array::from_vec(vec![1. as f64]).into_shape((1, 1)).unwrap();
        let out = network.feedforward(inp)[[0, 0]];

        let bias = network.biases[0][[0, 0]];
        let weight = network.weights[0][[0, 0]];

        let expected_out = Network::sigmoid(1. * weight + bias);

        assert_eq!(out, expected_out);

        // 2 inputs, 1 output
        let network = Network::new(vec![2, 1]);
        let inp = Array::from_vec(vec![1. as f64, 1. as f64])
            .into_shape((2, 1))
            .unwrap();
        let out = network.feedforward(inp)[[0, 0]];

        let bias = network.biases[0][[0, 0]];
        let weight_1 = network.weights[0][[0, 0]];
        let weight_2 = network.weights[0][[0, 1]];

        let expected_out = Network::sigmoid(1. * weight_1 + 1. * weight_2 + bias);

        assert_eq!(out, expected_out);

        // 1 input through 2 layers
        let network = Network::new(vec![1, 1, 1]);
        let inp = Array::from_vec(vec![1. as f64]).into_shape((1, 1)).unwrap();
        let out = network.feedforward(inp)[[0, 0]];

        let bias_1 = network.biases[0][[0, 0]];
        let bias_2 = network.biases[1][[0, 0]];
        let weight_1 = network.weights[0][[0, 0]];
        let weight_2 = network.weights[1][[0, 0]];

        let expected_out =
            Network::sigmoid(Network::sigmoid(1. * weight_1 + bias_1) * weight_2 + bias_2);

        assert_eq!(out, expected_out);
    }
}
