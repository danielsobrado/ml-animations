package nn

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"math/rand"
	"os"
	"strconv"
	"strings"
)

// GenerateXORData generates XOR training data.
func GenerateXORData() (*Tensor, *Tensor) {
	// XOR inputs: [0,0], [0,1], [1,0], [1,1]
	// XOR outputs: [0], [1], [1], [0]
	x := Zeros(4, 2)
	x.Set(0, 0, 0)
	x.Set(0, 1, 0)
	x.Set(1, 0, 0)
	x.Set(1, 1, 1)
	x.Set(2, 0, 1)
	x.Set(2, 1, 0)
	x.Set(3, 0, 1)
	x.Set(3, 1, 1)

	y := Zeros(4, 1)
	y.Set(0, 0, 0)
	y.Set(1, 0, 1)
	y.Set(2, 0, 1)
	y.Set(3, 0, 0)

	return x, y
}

// GenerateExpandedXORData generates expanded XOR data with noise.
func GenerateExpandedXORData(n int, rng *rand.Rand) (*Tensor, *Tensor) {
	x := Zeros(n, 2)
	y := Zeros(n, 1)

	for i := 0; i < n; i++ {
		a := rng.Intn(2)
		b := rng.Intn(2)
		xor := a ^ b

		// Add small noise to inputs
		noise := 0.1
		x.Set(i, 0, float64(a)+rng.Float64()*noise-noise/2)
		x.Set(i, 1, float64(b)+rng.Float64()*noise-noise/2)
		y.Set(i, 0, float64(xor))
	}

	return x, y
}

// TitanicData holds the preprocessed Titanic dataset.
type TitanicData struct {
	XTrain *Tensor
	YTrain *Tensor
	XTest  *Tensor
	YTest  *Tensor
}

// LoadTitanicData loads and preprocesses the Titanic dataset from CSV files.
func LoadTitanicData(trainPath, testPath string) (*TitanicData, error) {
	// Load training data
	trainRecords, err := readCSV(trainPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load train data: %w", err)
	}

	// Load test data
	testRecords, err := readCSV(testPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load test data: %w", err)
	}

	// Preprocess
	xTrain, yTrain := preprocessTitanic(trainRecords, true)
	xTest, yTest := preprocessTitanic(testRecords, true)

	return &TitanicData{
		XTrain: xTrain,
		YTrain: yTrain,
		XTest:  xTest,
		YTest:  yTest,
	}, nil
}

// readCSV reads a CSV file and returns records.
func readCSV(path string) ([][]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(bufio.NewReader(file))
	records := [][]string{}

	// Read header
	header, err := reader.Read()
	if err != nil {
		return nil, err
	}
	records = append(records, header)

	// Read data
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		records = append(records, record)
	}

	return records, nil
}

// preprocessTitanic preprocesses Titanic data.
// Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
func preprocessTitanic(records [][]string, hasLabel bool) (*Tensor, *Tensor) {
	if len(records) <= 1 {
		return Zeros(0, 7), Zeros(0, 1)
	}

	header := records[0]
	colIdx := make(map[string]int)
	for i, col := range header {
		colIdx[strings.ToLower(col)] = i
	}

	// Find column indices
	survivedIdx := colIdx["survived"]
	pclassIdx := colIdx["pclass"]
	sexIdx := colIdx["sex"]
	ageIdx := colIdx["age"]
	sibspIdx := colIdx["sibsp"]
	parchIdx := colIdx["parch"]
	fareIdx := colIdx["fare"]
	embarkedIdx := colIdx["embarked"]

	// Calculate stats for normalization
	var ages, fares []float64
	for _, record := range records[1:] {
		if age := parseFloat(record[ageIdx], -1); age >= 0 {
			ages = append(ages, age)
		}
		if fare := parseFloat(record[fareIdx], -1); fare >= 0 {
			fares = append(fares, fare)
		}
	}

	meanAge := mean(ages)
	stdAge := std(ages)
	meanFare := mean(fares)
	stdFare := std(fares)

	// Process records
	n := len(records) - 1
	x := Zeros(n, 7)
	y := Zeros(n, 1)

	validRows := 0
	for i, record := range records[1:] {
		// Skip incomplete records
		if len(record) < len(header) {
			continue
		}

		// Pclass (1, 2, 3) -> normalized
		pclass := parseFloat(record[pclassIdx], 2)
		x.Set(validRows, 0, (pclass-2)/1.0) // Center around 0

		// Sex (male=1, female=0)
		sex := 0.0
		if strings.ToLower(record[sexIdx]) == "male" {
			sex = 1.0
		}
		x.Set(validRows, 1, sex)

		// Age (normalized)
		age := parseFloat(record[ageIdx], meanAge)
		x.Set(validRows, 2, (age-meanAge)/stdAge)

		// SibSp
		sibsp := parseFloat(record[sibspIdx], 0)
		x.Set(validRows, 3, sibsp/3.0) // Rough normalization

		// Parch
		parch := parseFloat(record[parchIdx], 0)
		x.Set(validRows, 4, parch/3.0)

		// Fare (normalized)
		fare := parseFloat(record[fareIdx], meanFare)
		x.Set(validRows, 5, (fare-meanFare)/stdFare)

		// Embarked (S=0, C=1, Q=2) -> one value normalized
		embarked := 0.0
		switch strings.ToUpper(record[embarkedIdx]) {
		case "C":
			embarked = 0.5
		case "Q":
			embarked = 1.0
		}
		x.Set(validRows, 6, embarked)

		// Label
		if hasLabel {
			survived := parseFloat(record[survivedIdx], 0)
			y.Set(validRows, 0, survived)
		}

		validRows++
		_ = i
	}

	// Trim to valid rows
	if validRows < n {
		x = x.SliceRows(0, validRows)
		y = y.SliceRows(0, validRows)
	}

	return x, y
}

// parseFloat parses a string to float64, returning defaultVal on error.
func parseFloat(s string, defaultVal float64) float64 {
	s = strings.TrimSpace(s)
	if s == "" {
		return defaultVal
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return defaultVal
	}
	return v
}

// mean calculates the mean of a slice.
func mean(arr []float64) float64 {
	if len(arr) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range arr {
		sum += v
	}
	return sum / float64(len(arr))
}

// std calculates the standard deviation of a slice.
func std(arr []float64) float64 {
	if len(arr) == 0 {
		return 1
	}
	m := mean(arr)
	sum := 0.0
	for _, v := range arr {
		diff := v - m
		sum += diff * diff
	}
	variance := sum / float64(len(arr))
	if variance < 1e-10 {
		return 1
	}
	return sqrt(variance)
}

// sqrt is a simple square root implementation.
func sqrt(x float64) float64 {
	if x < 0 {
		return 0
	}
	z := x / 2
	for i := 0; i < 50; i++ {
		z = (z + x/z) / 2
	}
	return z
}
