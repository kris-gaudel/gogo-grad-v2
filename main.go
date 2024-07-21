package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

// _____ Structs _____

type Tensor struct {
	Data     mat.Dense
	Grad     mat.Dense
	Prev     []*Tensor
	Backward func()
}

type Batch struct {
	TensorArray  []*Tensor
	MaxBatchSize int
	CurrBatch    int
}

type Neuron struct {
	Weights     *Tensor
	Bias        *Tensor
	IsNonLinear bool
}

type Layer struct {
	Neurons []*Neuron
}

type Model struct {
	Layers []*Layer
}

type Trainer struct {
	Model        *Model
	Input        *Batch
	BatchSize    int
	Epochs       int
	LearningRate float64
}

// _____ Helper Functions _____
func EmptyMatrix(shape []int) mat.Dense {
	return *(mat.NewDense(shape[0], shape[1], nil))
}

// _____ Tensor Functions _____

// Initalize Tensors
func InitTensor(shape []int) *Tensor {
	if shape[0] <= 0 && shape[1] <= 0 {
		panic("Invalid shape!")
	}

	return &Tensor{
		Data:     *(mat.NewDense(shape[0], shape[1], nil)),
		Grad:     *(mat.NewDense(shape[0], shape[1], nil)),
		Prev:     nil,
		Backward: nil,
	}
}

func Random(shape []int) *Tensor {
	tensor := InitTensor(shape)
	tensor.Data.Apply(func(i, j int, v float64) float64 {
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		return r.Float64()*2 - 1
	}, &tensor.Data)
	return tensor
}

// Tensor Operations
func (t *Tensor) Add(other *Tensor) *Tensor {
	if t.Data.RawMatrix().Cols != other.Data.RawMatrix().Cols || t.Data.RawMatrix().Rows != other.Data.RawMatrix().Rows {
		// print tensor t
		fc := mat.Formatted(&t.Data, mat.Prefix(""), mat.Squeeze())
		fmt.Println(fc)

		// print tensor other
		fc = mat.Formatted(&other.Data, mat.Prefix(""), mat.Squeeze())
		fmt.Println(fc)
		panic("Invalid matrix addition!")
	}

	var newData mat.Dense
	newData.Add(&t.Data, &other.Data)

	out := InitTensor([]int{newData.RawMatrix().Rows, newData.RawMatrix().Cols})
	out.Data = newData
	out.Prev = []*Tensor{t, other}
	out.Backward = func() {
		var grad mat.Dense
		grad.Add(&t.Grad, &other.Grad)
		t.Grad.Add(&t.Grad, &grad)
		other.Grad.Add(&other.Grad, &grad)
	}
	return out
}

func (t *Tensor) Mult(other *Tensor) *Tensor {
	if t.Data.RawMatrix().Cols != other.Data.RawMatrix().Rows {
		// print tensor t
		fc := mat.Formatted(&t.Data, mat.Prefix(""), mat.Squeeze())
		fmt.Println(fc)

		// print tensor other
		fc = mat.Formatted(&other.Data, mat.Prefix(""), mat.Squeeze())
		fmt.Println(fc)
		panic("Invalid matrix multiplication!")
	}

	var newData mat.Dense
	newData.Mul(&t.Data, &other.Data)

	out := InitTensor([]int{newData.RawMatrix().Rows, newData.RawMatrix().Cols})
	out.Data = newData
	out.Prev = []*Tensor{t, other}
	out.Backward = func() {
		var gradT mat.Dense
		var otherT mat.Dense
		otherT.CloneFrom(other.Data.T())
		gradT.Mul(&out.Grad, &otherT)
		t.Grad.Add(&t.Grad, &gradT)

		var gradO mat.Dense
		var tensorT mat.Dense
		tensorT.CloneFrom(t.Data.T())
		gradO.Mul(&tensorT, &out.Grad)
		other.Grad.Add(&other.Grad, &gradO)
	}
	return out
}

func (t *Tensor) Tanh() *Tensor {
	var newData mat.Dense
	newData.Apply(func(i, j int, v float64) float64 {
		return math.Tanh(v)
	}, &t.Data)

	var newGrad mat.Dense
	newGrad.Apply(func(i, j int, v float64) float64 {
		return 1 - math.Pow(v, 2)
	}, &newData)

	out := InitTensor([]int{newData.RawMatrix().Rows, newData.RawMatrix().Cols})
	out.Data = newData
	out.Prev = []*Tensor{t}
	out.Backward = func() {
		var grad mat.Dense
		grad.MulElem(&newGrad, &out.Grad)
		t.Grad.Add(&t.Grad, &grad)
	}
	return out
}

func (t *Tensor) ReLU() *Tensor {
	var newData mat.Dense
	newData.Apply(func(i, j int, v float64) float64 {
		return math.Max(0, v)
	}, &t.Data)

	var newGrad mat.Dense
	newGrad.Apply(func(i, j int, v float64) float64 {
		if v > 0 {
			return 1
		}
		return 0
	}, &t.Data)

	out := InitTensor([]int{newData.RawMatrix().Rows, newData.RawMatrix().Cols})
	out.Data = newData
	out.Prev = []*Tensor{t}
	out.Backward = func() {
		var grad mat.Dense
		grad.MulElem(&newGrad, &out.Grad)
		t.Grad.Add(&t.Grad, &grad)
	}
	return out
}

// _____ Autograd Functions _____
func Backward(t *Tensor) {
	topo := []*Tensor{}
	visited := map[*Tensor]bool{}

	var buildTopo func(t *Tensor)
	buildTopo = func(t *Tensor) {
		if !visited[t] {
			visited[t] = true
			for _, prev := range t.Prev {
				buildTopo(prev)
			}
			topo = append(topo, t)
		}
	}
	buildTopo(t)

	t.Grad = EmptyMatrix([]int{t.Data.RawMatrix().Rows, t.Data.RawMatrix().Cols})
	for i := 0; i < t.Data.RawMatrix().Rows; i++ {
		for j := 0; j < t.Data.RawMatrix().Cols; j++ {
			t.Grad.Set(i, j, 1.0)
		}
	}

	for i := len(topo) - 1; i >= 0; i-- {
		topo[i].Backward()
	}
}

// _____ Batch Functions _____
func InitBatch(input *Tensor, batchSize int) *Batch {
	return &Batch{
		TensorArray:  []*Tensor{input},
		MaxBatchSize: batchSize,
		CurrBatch:    1,
	}
}

func EmptyBatch(batchSize int) *Batch {
	return &Batch{
		TensorArray:  []*Tensor{},
		MaxBatchSize: batchSize,
		CurrBatch:    0,
	}
}

func (b *Batch) AddToBatch(input *Tensor) {
	if b.CurrBatch == b.MaxBatchSize {
		panic("Batch is full!")
	}
	b.TensorArray = append(b.TensorArray, input)
	b.CurrBatch++
}

func (b *Batch) GetTensor(i int) *Tensor {
	if i >= len(b.TensorArray) {
		panic("Index out of bounds!")
	}
	return b.TensorArray[i]
}

// _____ Neuron Functions _____
func InitNeuron(inputSize int, isNonLinear bool) *Neuron {
	weights := Random([]int{inputSize, 1})
	bias := Random([]int{1, 1})

	return &Neuron{
		Weights:     weights,
		Bias:        bias,
		IsNonLinear: isNonLinear,
	}
}

func (n *Neuron) Forward(input *Tensor) *Tensor {
	weightedSum := input.Mult(n.Weights)
	weightedSum = weightedSum.Add(n.Bias)
	if n.IsNonLinear {
		return weightedSum.ReLU()
	}
	return weightedSum
}

// _____ Layer Functions _____
func InitLayer(inputSize, neuronCount int, isNonLinear bool) *Layer {
	neurons := make([]*Neuron, neuronCount)
	for i := 0; i < neuronCount; i++ {
		neurons[i] = InitNeuron(inputSize, isNonLinear)
	}
	return &Layer{
		Neurons: neurons,
	}
}

func (l *Layer) Forward(batch *Batch) *Batch {
	outputBatch := EmptyBatch(batch.MaxBatchSize)

	for _, input := range batch.TensorArray {
		var layerOutput *Tensor
		for _, neuron := range l.Neurons {
			neuronOutput := neuron.Forward(input)
			if layerOutput == nil {
				layerOutput = neuronOutput
			} else {
				layerOutput = layerOutput.Add(neuronOutput)
			}
		}
		outputBatch.AddToBatch(layerOutput)
	}
	return outputBatch
}

// _____ Model Functions _____
func InitModel(layers []*Layer) *Model {
	return &Model{
		Layers: layers,
	}
}

func (m *Model) Forward(batch *Batch) *Batch {
	for i, layer := range m.Layers {
		// Print layer
		fmt.Printf("Layer %d\n", i)
		for i, neuron := range layer.Neurons {
			// Print neuron
			fmt.Printf("Neuron %d\n", i)
			fout := mat.Formatted(&neuron.Weights.Data, mat.Prefix(""), mat.Squeeze())
			fmt.Println(fout)
			fout = mat.Formatted(&neuron.Bias.Data, mat.Prefix(""), mat.Squeeze())
			fmt.Println(fout)
		}
		fmt.Println("Done!")
		batch = layer.Forward(batch)
	}
	return batch
}

// _____ Main _____

func main() {
	// Create a batch of input tensors
	input1 := Random([]int{1, 3})
	input2 := Random([]int{1, 3})
	batch := InitBatch(input1, 2)
	batch.AddToBatch(input2)

	// Print the inputs of the batch
	for i, tensor := range batch.TensorArray {
		fmt.Printf("Input %d:\n", i)
		fout := mat.Formatted(&tensor.Data, mat.Prefix(""), mat.Squeeze())
		fmt.Println(fout)
	}

	// Initialize a layer
	layer := InitLayer(3, 2, true)

	// Initialize a model with the layer
	model := InitModel([]*Layer{layer})

	// Perform a forward pass
	outputBatch := model.Forward(batch)

	// Print the outputs of the batch
	for i, tensor := range outputBatch.TensorArray {
		fmt.Printf("Output %d:\n", i+1)
		fout := mat.Formatted(&tensor.Data, mat.Prefix(""), mat.Squeeze())
		fmt.Println(fout)
	}
}
