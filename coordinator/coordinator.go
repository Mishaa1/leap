// The main file for defining what a coordinator looks like.

package coordinator

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"flag"
	"github.com/rifflock/lfshook"
	"github.com/sirupsen/logrus"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/metadata"
	"io/ioutil"
	pb "leap/proto"
	"leap/sqlite"
	"leap/utils"
	"net"
	"os"
	"sync"
)

// A struct that holds the ip and port that the coordinator
// listens for requests from algorithms in the cloud, and the
// ip and port it listen for requests from algorithms in dis-
// tributed sites.
type Config struct {
	// The ip and port of the coordinator
	IpPort string
	// The ip and port of the cloud algos
	CloudAlgoIpPort string
	// Flag that determines whether to use SSL/TLS encryption
	Secure bool
	// File path to SSL/TLS certificate
	Crt string
	// File path to SSL/TLS key
	Key string
	// File path to the certificate authority crt
	CertAuth string
	// The common name of the site connector
	SiteConnCN string
	// The common name of the cloud algo
	CloudAlgoCN string
}

// Struct used to define a SiteConnector for the coordinator
type SiteConnector struct {
	// Id of a site connector
	id int64
	// Status where true = siteconn live and false = siteconn down
	available bool
	// Lock for site available
	availableMux sync.Mutex
	// Ip and port to contact this site connector
	ipPort string
}

type Coordinator struct {
	// Initial config
	Conf Config
	// Logging tool
	Log *logrus.Entry
	// The ID to be used for the next computation request that arrives
	ReqCounter int64
	// Lock to make counter access thread safe
	ReqCounterMux sync.Mutex
	// A concurrent map with site id as key and the ip and
	// port of the site as a value.
	SiteConnectors *utils.Map
	// Sqlite database
	Database *sqlite.Database
}

// Creates a new coordinator with the configurations given
// from the config struct.
//
// config: The ip and port configuration of the coordinator.
func NewCoordinator(config Config) *Coordinator {
	log := logrus.WithFields(logrus.Fields{"node": "coordinator"})
	return &Coordinator{Conf: config,
		Log:            log,
		ReqCounter:     0,
		SiteConnectors: utils.NewMap(),
		Database:       sqlite.CreateDatabase("leap-db", log),
	}
}

// Parses user flags and creates config using the given flags.
// If a flag is absent, use the default flag given in the
// config.json file.
//
// No args.
func GetConfig() Config {
	configPathPtr := flag.String("config", "../config/coord-config.json", "The path to the config file")
	flag.Parse()

	jsonFile, err := os.Open(*configPathPtr)
	if err != nil {
		logrus.WithFields(logrus.Fields{"node": "coordinator"}).Error("Could not find config file: " + *configPathPtr)
	}
	defer jsonFile.Close()
	jsonBytes, _ := ioutil.ReadAll(jsonFile)

	config := Config{}
	json.Unmarshal(jsonBytes, &config)

	return config
}

// Creates a 'Logs' directory if one doesn't exist, and creates
// a file to output the log files. This function also adds a
// hook to logrus, so that it can write to the file in text
// format, and display messages in terminal with colour.
//
// filepath: The path to the file where the logs should be added.
// dirpath:  The path to the directory where the logs will be located.
func AddFileHookToLogs(dirPath string) {
	_, err := os.Stat(dirPath)
	if os.IsNotExist(err) {
		os.Mkdir(dirPath, os.ModePerm)
	}

	filePath := dirPath + "coordinator.log"
	os.Create(filePath)

	hook := lfshook.NewHook(lfshook.PathMap{}, &logrus.JSONFormatter{})
	hook.SetDefaultPath(filePath)
	logrus.AddHook(hook)
}

// Creates a listener, registers the grpc server for the
// coordinator, and serves requests that arrive at the
// listener.
//
// No args.
func (c *Coordinator) Serve() {
	listener, err := net.Listen("tcp", c.Conf.IpPort)
	checkErr(c, err)
	c.Log.WithFields(logrus.Fields{"ip-port": c.Conf.IpPort}).Info("Listening for requests.")

	var s *grpc.Server
	if c.Conf.Secure {
		// Load coordinator certificates from disk
		cert, err := tls.LoadX509KeyPair(c.Conf.Crt, c.Conf.Key)
		if err != nil {
			c.Log.Error(err)
			return
		}

		// Create certificate pool from certificate authority
		certPool := x509.NewCertPool()
		ca, err := ioutil.ReadFile(c.Conf.CertAuth)
		if err != nil {
			c.Log.Error(err)
			return
		}

		// Append client certificates from certificate authority
		ok := certPool.AppendCertsFromPEM(ca)
		if !ok {
			c.Log.Error("Error when appending client certs")
		}

		// Create TLS credentials
		creds := credentials.NewTLS(&tls.Config{
			ClientAuth:   tls.RequireAndVerifyClientCert,
			Certificates: []tls.Certificate{cert},
			ClientCAs:    certPool,
		})

		opts := []grpc.ServerOption{
			grpc.Creds(creds),
			grpc.UnaryInterceptor(authenticate),
		}

		s = grpc.NewServer(opts...)

	} else {

		opts := []grpc.ServerOption{
			grpc.UnaryInterceptor(authenticate),
		}

		s = grpc.NewServer(opts...)
	}

	pb.RegisterCoordinatorServer(s, c)
	err = s.Serve(listener)
	checkErr(c, err)
}

// This function does basically the same job as grpc dial,
// but it loads the proper credentials and establishes a
// secure connection if the secure flag is turned on.
//
// addr: The address where you want to establish a connection
// serverName: The common name of the server to be contacted
func (c *Coordinator) Dial(addr string, servername string) (*grpc.ClientConn, error) {
	if c.Conf.Secure {
		cert, err := tls.LoadX509KeyPair(c.Conf.Crt, c.Conf.Key)
		checkErr(c, err)

		certPool := x509.NewCertPool()
		ca, err := ioutil.ReadFile(c.Conf.CertAuth)
		checkErr(c, err)

		certPool.AppendCertsFromPEM(ca)
		creds := credentials.NewTLS(&tls.Config{
			ServerName:   servername,
			Certificates: []tls.Certificate{cert},
			RootCAs:      certPool,
		})

		return grpc.Dial(addr, grpc.WithTransportCredentials(creds))

	} else {
		return grpc.Dial(addr, grpc.WithInsecure())
	}
}

func authenticate(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo,
	handler grpc.UnaryHandler) (interface{}, error) {

	if info.FullMethod == "/proto.Coordinator/Compute" {
		_, ok := metadata.FromIncomingContext(ctx)

		if !ok {
			return nil, errors.New("Missing metadata.")
		}
	}

	return handler(ctx, req)
}

// TODO: Add request id to checkErr
// Helper to log errors in the coordinator.
//
// coord: Coordinator instance (holds logging tool)
// err: Error returned by a function that should be checked
//      if nil or not.
func checkErr(c *Coordinator, err error) {
	if err != nil {
		c.Log.Error(err.Error())
	}
}
