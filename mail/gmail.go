package mail

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"net/mail"
	"net/url"
	"os"
	"os/user"
	"path/filepath"
	"strings"
	"time"

	"golang.org/x/net/context"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"google.golang.org/api/gmail/v1"
	"google.golang.org/api/option"
)

// GetClient uses a Context and Config to retrieve a Token
// then generate a Client. It returns the generated Client.
func GetClient(ctx context.Context, config *oauth2.Config, code string) *http.Client {
	cacheFile, err := TokenCacheFile()
	if err != nil {
		log.Fatalf("Unable to get path to cached credential file. %v", err)
	}
	tok, err := TokenFromFile(cacheFile)
	if err != nil {
		if code == "" {
			GetTokenFromWeb(config)
			return nil
		}
		tok = GetTokenFromCode(code, config)
		SaveToken(cacheFile, tok)
	}
	return config.Client(ctx, tok)
}

// GetTokenFromWeb uses Config to request a Token.
// It returns the retrieved Token.
func GetTokenFromWeb(config *oauth2.Config) {
	authURL := config.AuthCodeURL("state-token", oauth2.AccessTypeOffline)
	fmt.Printf("Go to the following link in your browser then type the "+
		"authorization code: \n%v\n", authURL)
	os.Exit(1)
}
func GetTokenFromCode(code string, config *oauth2.Config) *oauth2.Token {

	tok, err := config.Exchange(context.TODO(), code)
	if err != nil {
		log.Fatalf("Unable to retrieve token from web %v", err)
	}
	return tok
}

// TokenCacheFile generates credential file path/filename.
// It returns the generated credential path/filename.
func TokenCacheFile() (string, error) {
	usr, err := user.Current()
	if err != nil {
		return "", err
	}
	tokenCacheDir := filepath.Join(usr.HomeDir, ".credentials")
	os.MkdirAll(tokenCacheDir, 0700)
	return filepath.Join(tokenCacheDir,
		url.QueryEscape("gmail-go-quickstart.json")), err
}

// TokenFromFile retrieves a Token from a given file path.
// It returns the retrieved Token and any read error encountered.
func TokenFromFile(file string) (*oauth2.Token, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	t := &oauth2.Token{}
	err = json.NewDecoder(f).Decode(t)
	defer f.Close()
	return t, err
}

// SaveToken uses a file path to create a file and store the
// token in it.
func SaveToken(file string, token *oauth2.Token) {
	fmt.Printf("Saving credential file to: %s\n", file)
	f, err := os.Create(file)
	if err != nil {
		log.Fatalf("Unable to cache oauth token: %v", err)
	}
	defer f.Close()
	json.NewEncoder(f).Encode(token)
}

// Connect establishes a connection to Gmail API and returns a service instance
func Connect(pathToSecret string, code string) (*gmail.Service, error) {
	ctx := context.Background()

	b, err := os.ReadFile(pathToSecret)
	if err != nil {
		log.Printf("Unable to read client secret file: %v\n", err)
		return nil, err
	}

	config, err := google.ConfigFromJSON(b, gmail.GmailReadonlyScope)
	if err != nil {
		log.Printf("Unable to parse client secret file to config: %v\n", err)
		return nil, err
	}
	client := GetClient(ctx, config, code)

	srv, err := gmail.NewService(ctx, option.WithHTTPClient(client))
	if err != nil {
		log.Printf("Unable to create gmail Client %v\n", err)
		return nil, err
	}

	return srv, nil
}

// GetEmails retrieves emails from Gmail with the specified parameters and processes them using the provided callback
func GetEmails(srv *gmail.Service, maxId string, filterLabel string, processor func(email Email)) (error, string) {
	maxReached := false
	msgid := maxId
	firstId := true
	user := "me"
	pageToken := ""

	expireTime := time.Now().Add(DEFAULT_TIME)
	log.Println("expiry time for emails is ", expireTime)
	log.Println("maxid requested is ", maxId)

	for {
		req := srv.Users.Messages.List(user).LabelIds(filterLabel)
		if pageToken != "" {
			req.PageToken(pageToken)
		}
		msgs, err := req.Do()
		if err != nil {
			log.Printf("error reading emails %v\n", err)
			return err, ""
		}
		if msgs == nil {
			log.Println("error reading msgs")
			return err, ""
		}

		for _, mpart := range msgs.Messages {
			email := Email{Id: mpart.Id}
			if firstId {
				msgid = mpart.Id
				firstId = false
			}

			m, err := srv.Users.Messages.Get(user, mpart.Id).Do()
			if err != nil {
				log.Printf("error reading emails %v\n", err)
				return err, ""
			}

			email.LabelIds = m.LabelIds

			for _, h := range m.Payload.Headers {
				if strings.Contains(h.Name, "Subject") {
					email.Subject = h.Value
					fmt.Println("Subject" + email.Subject)
				}
				if strings.Compare(h.Name, "From") == 0 {
					email.From, err = GetEmailAddress(h.Value)
					if err != nil {
						log.Printf("Error getting address %v\n", err)
						email.From = &mail.Address{Name: h.Value}
					}
				}
				if strings.Compare(h.Name, "To") == 0 {
					email.To, err = GetEmailAddress(h.Value)
					if err != nil {
						log.Printf("Error getting address %v\n", err)
						email.To = &mail.Address{Name: h.Value}
					}
				}
				if strings.Contains(h.Name, "Date") {
					email.Date = strings.Split(h.Value, "(")[0]
					email.Date = strings.TrimSpace(email.Date)
				}
				if strings.Contains(h.Name, "Content-Type") {
					email.Type = h.Value
				}
			}

			if !strings.Contains(maxId, NO_MAX_ID) {
				if strings.Compare(email.Id, maxId) <= 0 {
					maxReached = true
					log.Println("found maxId")
					break
				}
			} else {
				emailTime, err := time.Parse(DATE_PARSING, email.Date)
				if err == nil && expireTime.After(emailTime) {
					maxReached = true
					log.Println("maximum time reached ", emailTime)
					break
				}
				if err != nil {
					log.Println("error occurred parsing email time", err)
				}
			}

			//need to extract normal text and apply
			ProcessPart(&email, m.Payload)
			if len(email.BodyHtml) > 0 {
				email.BodyHtml = EncodeBase64(email.BodyHtml)
			}
			if len(email.Body) > 0 {
				email.Body = EncodeBase64(email.Body)
			}
			email.IsFwd, _ = IsForward(email.Subject)
			email.IsRE, _ = IsReply(email.Subject)

			if email.IsFwd {
				email.EmbeddedAddresses = GetEmbeddedAddress(email.Text())
			}
			if email.From != nil {
				go processor(email)
			} else {
				log.Printf("Error processing address\n")
			}
		}

		if msgs.NextPageToken == "" || maxReached {
			break
		}
		pageToken = msgs.NextPageToken
		log.Println("page processed emails -", len(msgs.Messages))
	}
	log.Println("new max id is", msgid)

	return nil, msgid
}

// GetIdForLabel retrieves the label ID for a given label name
func GetIdForLabel(srv *gmail.Service, labelName string) (string, error) {
	user := "me"
	r, err := srv.Users.Labels.List(user).Do()
	if err != nil {
		log.Printf("Unable to retrieve labels. %v\n", err)
		return "", err
	}
	for _, label := range r.Labels {
		if label.Name == labelName {
			log.Printf("filter label %s has id %s \n", labelName, label.Id)
			return label.Id, nil
		}
	}
	return "", errors.New("Label not found")
}
