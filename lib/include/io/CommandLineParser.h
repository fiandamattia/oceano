/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2022 - 2026 by CNR-ISMAR
 *
 * This code, as the deal.II library is free software; you can use it,
 * redistribute it, and/or modify it under the terms of the GNU Lesser
 * General Public License as published by the Free Software Foundation;
 * either version 2.1 of the License, or (at your option) any later
 * version. The full text of the license can be found in the file
 * LICENSE.md at the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Luca Arpaia, 2023
 *         Giuseppe Orlando, 2026
 */
#ifndef COMMANDLINEPARSER_H
#define COMMANDLINEPARSER_H

/**
 * Namespace containing the input/output file handler.
 */

namespace IO
{

  using namespace dealii; // usual namespace for deal.II programs


  
  // The next class is responsible for parsing correctly the command line.
  // parsing means to check that the user has provided the correct number of arguments and
  // that the arguments are correct.
  class CommandLineParser
  {
  public: // recall: public members are accessible from outside the class
    CommandLineParser() = default;
    ~CommandLineParser() = default;

    void parse_command_line(int argc, char **argv);

  private: // recall: private members are accessible only from inside the class
    void usage();
  };



  void CommandLineParser::usage() // usage is a private method containing the "error usage" message to be printed in case of wrong command line usage
  {
    std::cout << "Oceano usage:" << std::endl;
    std::cout << " -i   --input <file>    :"
              << " use a configuration file in deal.II PRM format as input."
              << std::endl;
    std::cout << " -h   --help            :" << " print this message "
              << std::endl;
  }

  // The function `parse_command_line` does not output anything. It just
  // raises messages errors and exit the program in case of wrong usage
  // of the executable.
  // We expect three arguments: the name of the executable, the option -i or --input and the name of the configuration file.
  // index start from 0, so the first argument is argv[0], the second is argv[1] and the third is argv[2].
  void CommandLineParser::parse_command_line(int argc, char **argv)
  {
    if (argc == 1) // only the name of the executable is provided, no other arguments are given
      {
        usage();
        AssertThrow(false,
                    ExcMessage("No configuration file was given.\n"
                               "Check again your command line."));
      }
    std::string option(argv[1]); // argument 1 is the option, either -i or --input or -h or --help
    if ( option == "-h" || option == "--help" )
      {
        usage();
        std::exit(1);
      }
    else if ( option == "-i" || option == "--input" )
      {
        if (argc == 2) // the option -i or --input is given, but no configuration file is provided
          {
            usage();
            AssertThrow(false,
                        ExcMessage("No configuration file was given after -i\n"
                                   "Check again your command line."));
         }
      }
    else
      {
        usage();
        AssertThrow(false,
                    ExcMessage("It was not possible to read the "
                               "configuration file.\n"
                               "Check again your command line."));
      }
  }

} // namespace IO
#endif //COMMANDLINEPARSER_H
