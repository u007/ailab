package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"code.sajari.com/docconv"
	"github.com/google/generative-ai-go/genai"
	"github.com/joho/godotenv"
	"google.golang.org/api/option"
)

type Claim struct {
	Date       time.Time
	BillerName string
	ClaimType  string
	Amount     float64
	Currency   string
}

// callGeminiWithRetry attempts to call the Gemini API with retries and exponential backoff.
func callGeminiWithRetry(ctx context.Context, model *genai.GenerativeModel, prompt string, filePath string) (*genai.GenerateContentResponse, error) {
	const maxRetries = 5
	initialDelay := 5 * time.Second

	for i := 0; i < maxRetries; i++ {
		resp, err := model.GenerateContent(ctx, genai.Text(prompt))
		if err == nil {
			return resp, nil
		}

		log.Printf("Error generating content for file %s (attempt %d/%d): %v", filePath, i+1, maxRetries, err)

		// Check for quota error (Error 429)
		if strings.Contains(err.Error(), "googleapi: Error 429") {
			delay := initialDelay * time.Duration(1<<uint(i)) // Exponential backoff
			log.Printf("Retrying in %v...", delay)
			time.Sleep(delay)
			continue
		} else {
			// For other errors, do not retry
			return nil, err
		}
	}
	return nil, fmt.Errorf("failed to generate content for file %s after %d retries", filePath, maxRetries)
}

// parseDate attempts to parse a date string using multiple common layouts.
func parseDate(dateStr string) (time.Time, error) {
	layouts := []string{
		"2006-01-02",         // YYYY-MM-DD
		"January 2, 2006",    // Month Day, Year
		"Jan 2, 2006",        // Abbreviated Month Day, Year
		"2 January 2006",     // Day FullMonth Year
		"2 Jan 2006",         // Day AbbreviatedMonth Year
		"02.01.2006",         // DD.MM.YYYY
		"2006/01/02",         // YYYY/MM/DD
		"01/02/2006",         // MM/DD/YYYY
		"02/01/2006",         // DD/MM/YYYY
		"2006-01-02T15:04:05Z07:00", // RFC3339
	}

	for _, layout := range layouts {
		t, err := time.Parse(layout, dateStr)
		if err == nil {
			return t, nil
		}
	}
	return time.Time{}, fmt.Errorf("could not parse date: %s with any known layouts", dateStr)
}

func main() {
	filePath := flag.String("path", "", "Path to a file or folder to process")
	apiKey := flag.String("api-key", "", "Google AI API Key")
	restartProcess := flag.Bool("restart", false, "Start processing from scratch, ignoring processed_files.log and output.csv")
	flag.Parse()

	if err := godotenv.Load(); err != nil {
		log.Println("No .env file found, using command-line flags or environment variables")
	}

	if *apiKey == "" {
		*apiKey = os.Getenv("GOOGLE_API_KEY")
	}

	if *filePath == "" {
		log.Fatal("Please provide a path to a file or folder to process")
	}
	if *apiKey == "" {
		log.Fatal("Please provide a Google AI API Key via the --api-key flag or a .env file")
	}

	// Handle restart option
	processedLogFile := "processed_files.log"
	outputCsvFile := "output.csv"
	if *restartProcess {
		log.Println("Restarting process: Deleting processed_files.log and output.csv")
		if err := os.Remove(processedLogFile); err != nil && !os.IsNotExist(err) {
			log.Printf("Warning: Could not delete processed_files.log: %v", err)
		}
		if err := os.Remove(outputCsvFile); err != nil && !os.IsNotExist(err) {
			log.Printf("Warning: Could not delete output.csv: %v", err)
		}
	}

	ctx := context.Background()
	client, err := genai.NewClient(ctx, option.WithAPIKey(*apiKey))
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	model := client.GenerativeModel("gemini-2.5-flash")

	var claims []Claim
	var totalTokens int64
	var processedFiles, failedFiles int

	// Resumability: Load already processed files
	processedFilePaths := make(map[string]bool)
	if _, err := os.Stat(processedLogFile); err == nil {
		content, err := os.ReadFile(processedLogFile)
		if err != nil {
			log.Printf("Warning: Could not read processed_files.log: %v", err)
		} else {
			lines := strings.Split(string(content), "\n")
			for _, line := range lines {
				trimmedLine := strings.TrimSpace(line)
				if trimmedLine != "" {
					processedFilePaths[trimmedLine] = true
				}
			}
			log.Printf("Loaded %d previously processed files from %s", len(processedFilePaths), processedLogFile)
		}
	}

	// Open processed_files.log for appending
	processedLog, err := os.OpenFile(processedLogFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatalf("Failed to open processed_files.log: %v", err)
	}
	defer processedLog.Close()

	err = filepath.Walk(*filePath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if !info.IsDir() {
			// Check if file has already been processed
			if processedFilePaths[path] {
				fmt.Printf("Skipping already processed file: %s\n", path)
				return nil
			}

			switch filepath.Ext(path) {
			case ".pdf", ".docx":
				fmt.Printf("Processing file: %s\n", path)
				content, err := docconv.ConvertPath(path)
				if err != nil {
					log.Printf("Error converting file %s: %v", path, err)
					failedFiles++
					return nil
				}

				prompt := fmt.Sprintf("Extract the following information from the document: date, biller name, claim type, amount, and currency. If currency is not specified, default to MYR. Provide the output in JSON format with the keys: 'date', 'biller_name', 'claim_type', 'amount', 'currency'. Document content: %s", content)

				resp, err := callGeminiWithRetry(ctx, model, prompt, path) // Use the retry function
				if err != nil {
					log.Printf("Failed to get response for file %s after retries: %v", path, err)
					failedFiles++
					return nil
				}

				if resp.UsageMetadata != nil {
					totalTokens += int64(resp.UsageMetadata.TotalTokenCount)
				}

				if len(resp.Candidates) > 0 {
					var result map[string]interface{}
					// Clean the response to be valid JSON
					jsonString := strings.Trim(fmt.Sprintf("%s", resp.Candidates[0].Content.Parts[0]), "` \n")
					jsonString = strings.TrimPrefix(jsonString, "json")
					jsonString = strings.TrimSpace(jsonString)

					if err := json.Unmarshal([]byte(jsonString), &result); err != nil {
						log.Printf("Error unmarshalling JSON for file %s: %v", path, err)
						failedFiles++
						return nil
					}

					date, dateErr := parseDate(result["date"].(string)) // Use the new parseDate function
					if dateErr != nil {
						log.Printf("Error parsing date for file %s: %v", path, dateErr)
						failedFiles++
						return nil
					}

					amount, amountErr := strconv.ParseFloat(fmt.Sprintf("%v", result["amount"]), 64)
					if amountErr != nil {
						log.Printf("Error parsing amount for file %s: %v", path, amountErr)
						failedFiles++
						return nil
					}

					// Ensure biller_name, claim_type, currency are strings, providing defaults if necessary
					billerName, ok := result["biller_name"].(string)
					if !ok {
						billerName = "UNKNOWN_BILLER"
						log.Printf("Warning: biller_name not a string for file %s, defaulting to %s", path, billerName)
					}
					claimType, ok := result["claim_type"].(string)
					if !ok {
						claimType = "UNKNOWN_CLAIM_TYPE"
						log.Printf("Warning: claim_type not a string for file %s, defaulting to %s", path, claimType)
					}
					currency, ok := result["currency"].(string)
					if !ok {
						currency = "MYR" // Default as per prompt
						log.Printf("Warning: currency not a string for file %s, defaulting to %s", path, currency)
					}

					claims = append(claims, Claim{
						Date:       date,
						BillerName: billerName,
						ClaimType:  claimType,
						Amount:     amount,
						Currency:   currency,
					})
					processedFiles++

					// Mark file as processed
					_, err = processedLog.WriteString(path + "\n")
					if err != nil {
						log.Printf("Error writing to processed_files.log: %v", err)
					}
					processedFilePaths[path] = true // Add to in-memory map for current run
				} else {
					log.Printf("No candidates returned from Gemini API for file %s", path)
					failedFiles++
				}
			}
		}
		return nil
	})

	if err != nil {
		log.Fatalf("Error processing files: %v", err)
	}

	sort.Slice(claims, func(i, j int) bool {
		return claims[i].Date.Before(claims[j].Date)
	})

	// Append to output.csv instead of overwriting, if it exists
	csvFile, err := os.OpenFile(outputCsvFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatalf("failed opening/creating output.csv: %s", err)
	}
	defer csvFile.Close()

	writer := csv.NewWriter(csvFile)
	defer writer.Flush()

	// Write header only if the file is new/empty
	stat, err := csvFile.Stat()
	if err != nil {
		log.Fatalf("failed getting output.csv stat: %s", err)
	}
	if stat.Size() == 0 {
		writer.Write([]string{"Date", "Biller Name", "Claim Type", "Amount", "Currency"})
	}

	for _, claim := range claims {
		writer.Write([]string{
			claim.Date.Format("2006-01-02"),
			claim.BillerName,
			claim.ClaimType,
			fmt.Sprintf("%.2f", claim.Amount),
			claim.Currency,
		})
	}

	fmt.Println("\n--- Processing Summary ---")
	fmt.Printf("Total files processed (this run): %d\n", processedFiles)
	fmt.Printf("Total files failed (this run): %d\n", failedFiles)
	fmt.Printf("Total token usage (this run): %d\n", totalTokens)
	fmt.Printf("Total files previously processed: %d\n", len(processedFilePaths)-processedFiles) // Adjust for files processed in this run
	fmt.Println("\nProcessing complete. Output written to output.csv and processed files logged to processed_files.log")
}
