package mail

import (
	"encoding/base64"
	"net/mail"
	"regexp"
	"strings"
	"time"

	"golang.org/x/net/html"
	"google.golang.org/api/gmail/v1"
)

const (
	TYPE_MULTIPART  = "multipart"
	TYPE_TEXT       = "text"
	TYPE_TEXT_PLAIN = "text/plain"
	TYPE_TEXT_HTML  = "text/html"
	DEFAULT_TIME    = -1 * 24 * time.Hour
	DEVICE_PREFIX   = "gmail_"
	DATE_PARSING    = "Mon, 2 Jan 2006 15:04:05 -0700"
	NO_MAX_ID       = "NO_ID"
	UPDATE_CYCLE    = 3600
)

type Email struct {
	Id                string
	From              *mail.Address
	To                *mail.Address
	Date              string
	Type              string
	Subject           string
	Body              string
	BodyHtml          string
	IsFwd             bool
	IsRE              bool
	EmbeddedAddresses []string
	LabelIds          []string
}

var EN_FWD = []string{"fw:", "fwd:", "fwd ", "fw "}
var EN_RE = []string{"re:", "re ", "re(", "re["}

func (e *Email) Text() string {
	if len(e.Body) > 0 {
		res, _ := DecodeBase64(e.Body)
		return res
	} else {
		bd, err := DecodeBase64(e.BodyHtml)
		if err == nil {
			out, _ := HtmlToText(bd)
			return out
		}
	}
	return ""
}

func DecodeBase64(body string) (string, error) {
	data, err := base64.StdEncoding.DecodeString(body)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func EncodeBase64(body string) string {
	return base64.StdEncoding.EncodeToString([]byte(body))
}

// checks if the email title has a fordward prefix
func IsForward(title string) (bool, error) {
	title = strings.ToLower(strings.TrimSpace(title))
	var fwds []string
	fwds = EN_FWD

	for _, f := range fwds {
		if strings.HasPrefix(title, f) {
			return true, nil
		}
	}
	return false, nil
}

func IsReply(title string) (bool, error) {
	title = strings.ToLower(strings.TrimSpace(title))
	var fwds []string
	fwds = EN_RE

	for _, f := range fwds {
		if strings.HasPrefix(title, f) {
			return true, nil
		}
	}
	return false, nil
}

func GetEmailAddress(t string) (*mail.Address, error) {
	return mail.ParseAddress(t)
}

func HtmlToText(h string) (string, error) {
	out := ""
	doc, err := html.Parse(strings.NewReader(h))
	if err != nil {
		return out, err
	}
	var f func(*html.Node, *string)
	f = func(n *html.Node, out *string) {
		if n.Type == html.TextNode {
			*out += n.Data
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			f(c, out)
		}
	}
	f(doc, &out)
	return out, nil

}
func GetEmbeddedAddress(t string) []string {
	e := regexp.MustCompile("(^|\\S|<)\\S+@.+\\.\\S+(>|$)")
	res := e.FindAllString(t, -1)
	for i, s := range res {
		s = strings.Replace(s, "<", "", -1)
		res[i] = strings.Replace(s, ">", "", -1)
	}

	return res
}

func DecodeGmailBody(body string) string {
	body = strings.Replace(body, "-", "+", -1)
	body = strings.Replace(body, "_", "/", -1)
	return body
}

func ProcessPart(mail *Email, part *gmail.MessagePart) {
	if strings.HasPrefix(part.MimeType, TYPE_TEXT) {
		switch part.MimeType {
		case TYPE_TEXT_HTML:
			res, _ := DecodeBase64(DecodeGmailBody(part.Body.Data))
			mail.BodyHtml += res
			return
		default:
			res, _ := DecodeBase64(DecodeGmailBody(part.Body.Data))
			mail.Body += res
			return
		}
	} else if strings.HasPrefix(part.MimeType, TYPE_MULTIPART) {
		for _, p := range part.Parts {
			ProcessPart(mail, p)
		}
	}

}
