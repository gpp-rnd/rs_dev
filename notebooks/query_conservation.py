import requests
import pandas as pd
import sys


def get_chr_from_refseq(refseq):
    num = int(refseq[-5:-3])
    if num == 23:
        return 'chrX'
    elif num == 24:
        return 'chrY'
    else:
        return 'chr' + str(num)


def get_conservation(refseq, start, end):
    chrom = get_chr_from_refseq(refseq)
    api_url = 'http://api.genome.ucsc.edu/getData/track'
    params = {
        'genome': 'hg38',
        'track': 'phyloP100way',
        'start': start,
        'end': end,
        'chrom': chrom
    }
    results = requests.get(api_url, data=params)
    if results.ok:
        value_df = (pd.DataFrame([pd.Series(x) for x in pd.read_json(results.content.decode('utf8'))[chrom].values])
                    .rename(columns={'value': 'conservation'}))
    else:
        raise ValueError(results.reason)
    return value_df


def build_request_url(ext, server="https://rest.ensembl.org"):
    request_url = "/".join([server, ext])
    return request_url


def handle_results(results):
    if not results.ok:
        results.raise_for_status()
        sys.exit()
    decoded = results.json()
    return decoded


def get_transcript_exons(transcript):
    base_transcript = transcript.split('.')[0]
    request_url = build_request_url("/lookup/id/" + base_transcript + "?expand=1")
    r = requests.get(request_url, headers={"Content-Type": "application/json"})
    decoded = handle_results(r)
    exon_df = pd.DataFrame(decoded['Exon'])
    return exon_df


def get_exon_conservation(exon_df, refseq_id, intron_depth):
    conservation_dict = {}
    for i, row in exon_df.set_index('id').iterrows():
        conservation_dict[i] = get_conservation(refseq_id, row['start'], row['end'])
        # get the conservation of introns flanking the exon
        conservation_dict[i + '_preceding_intron'] = get_conservation(refseq_id, row['start'] - intron_depth,
                                                                      row['start'])
        conservation_dict[i + '_succeeding_intron'] = get_conservation(refseq_id, row['end'],
                                                                       row['end'] + intron_depth)
    conservation_df = (pd.concat(conservation_dict)
                       .reset_index(level=0)
                       .reset_index(drop=True)
                       .rename({'level_0': 'exon_id'}, axis=1))
    return conservation_df


def get_transcript_conservation(transcript_id, refseq_id, intron_depth=30):
    """ Get conservation of each exon of a transcript

    Parameters
    ----------
    transcript_id

    Returns
    -------

    """
    exon_df = get_transcript_exons(transcript_id)
    conservation_df = get_exon_conservation(exon_df, refseq_id, intron_depth)
    return conservation_df
