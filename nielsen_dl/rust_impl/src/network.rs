use rand::distributions::{Normal, IndependentSample};
use rand;

use na::{Dynamic, MatrixVec, Matrix};


type BiasVectors = Vec<Vec<f64>>;
type WeightMatrix = Matrix<f64, Dynamic, Dynamic, MatrixVec<f64, Dynamic, Dynamic>>;

pub struct Network {
    num_layers: usize,
    sizes: Vec<u32>,

    biases: BiasVectors,
    // A weight matrix is stored for each level of the network
    weights: Vec<WeightMatrix>,
}

impl Network {
    // Sizes specifies the number of neurons in each layer
    pub fn new(sizes: Vec<u32>) -> Network {
        let mut rng = rand::thread_rng();
        let std_normal = Normal::new(0.0, 1.0);

        let num_layers = sizes.len();

        let mut biases: BiasVectors = Vec::with_capacity(num_layers - 1);
        for layer in sizes.iter().skip(1) {
            let mut layer_biases: Vec<f64> = Vec::with_capacity(*layer as usize);

            for _ in 0..*layer {
                layer_biases.push(std_normal.ind_sample(&mut rng));
            }

            biases.push(layer_biases);
        }

        let mut weights: Vec<WeightMatrix> = Vec::with_capacity(num_layers - 1);
        for (i,j) in sizes.iter().zip(sizes.iter().skip(1)) {
            let mut pair_weights: Vec<f64> = Vec::with_capacity((i * j) as usize);
            for _ in 0..(i * j) {
                pair_weights.push(std_normal.ind_sample(&mut rng));
            }

            // weights.push(Matrix::new(*i as usize, *j as usize, pair_weights));
        }


        unimplemented!()
    }
}
